"""
Extended model suite for overnight sweep:
  Linear, Validation-tuned Ridge, LightGBM, CUDA XGBoost,
  GRU, LSTM, BiLSTM, TCN, CNN-LSTM, and NNLS stack.
Returns raw/scaled validation & test predictions plus state dicts for best checkpoint saving.
"""
import time

import numpy as np


def fit_linear_and_ridge(Xtr, ytr, Xva, va_y_raw, inv_va, Xte, inv_te):
    from sklearn.linear_model import LinearRegression, Ridge

    lin = LinearRegression()
    lin.fit(Xtr, ytr)
    lin_va = inv_va(lin.predict(Xva))
    lin_te = inv_te(lin.predict(Xte))

    best_alpha, best_rmse, best_rva, best_rte, best_r = None, float("inf"), None, None, None
    for a in (0.1, 1.0, 3.2, 10.0, 32.0, 100.0, 1000.0):
        m = Ridge(alpha=a)
        m.fit(Xtr, ytr)
        pva = inv_va(m.predict(Xva))
        r = float(np.sqrt(np.mean((pva - va_y_raw) ** 2)))
        if r < best_rmse:
            best_rmse, best_alpha, best_rva, best_r = r, a, pva, m
    best_rte = inv_te(best_r.predict(Xte))
    return (lin_va, lin_te), (best_rva, best_rte, best_alpha)


def fit_gbm(Xtr, ytr, Xva, yva, Xte, seed=0):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.05, num_leaves=63,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        n_jobs=-1, random_state=seed, verbose=-1,
    )
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m.predict(Xva), m.predict(Xte), m


def fit_xgb(Xtr, ytr, Xva, yva, Xte, seed=0):
    import torch
    import xgboost as xgb
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = xgb.XGBRegressor(
        n_estimators=2000, learning_rate=0.05, max_depth=7, subsample=0.8,
        colsample_bytree=0.8, early_stopping_rounds=50, random_state=seed,
        device=dev, tree_method="hist",
    )
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m.predict(Xva), m.predict(Xte), m


def build_torch_model(kind, c_in, hidden=64):
    import torch
    import torch.nn as nn

    class RNN(nn.Module):
        def __init__(self, k, bi):
            super().__init__()
            self.bi, self.hidden = bi, hidden
            self.rnn = (nn.GRU if k == "gru" else nn.LSTM)(c_in, hidden, batch_first=True, bidirectional=bi)
            self.head = nn.Linear(hidden * (2 if bi else 1), 1)

        def forward(self, x):
            o, _ = self.rnn(x)
            if self.bi:
                z = torch.cat([o[:, -1, :self.hidden], o[:, 0, self.hidden:]], dim=1)
            else:
                z = o[:, -1]
            return self.head(z).squeeze(-1)

    class TCN(nn.Module):
        def __init__(self, k=3, dil=(1, 2, 4)):
            super().__init__()
            layers, c = [], c_in
            for d in dil:
                layers += [nn.ConstantPad1d(((k - 1) * d, 0), 0.0), nn.Conv1d(c, hidden, k, dilation=d), nn.ReLU()]
                c = hidden
            self.net = nn.Sequential(*layers)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            z = self.net(x.transpose(1, 2))
            return self.head(z[:, :, -1]).squeeze(-1)

    class CNNLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(c_in, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            # x: (B, L, C) -> Conv1d expects (B, C, L) -> back to (B, L, H)
            z = self.conv(x.transpose(1, 2)).transpose(1, 2)
            o, _ = self.lstm(z)
            return self.head(o[:, -1]).squeeze(-1)

    if kind == "gru":
        return RNN("gru", False)
    if kind == "lstm":
        return RNN("lstm", False)
    if kind == "bilstm":
        return RNN("lstm", True)
    if kind == "tcn":
        return TCN()
    if kind == "cnn_lstm":
        return CNNLSTM()
    raise ValueError(kind)


def train_neural(kind, Xtr, ytr, Xva, yva, Xte, seed, epochs=60, patience=7, lr=1e-3,
                 batch=2048, hidden=64, use_bf16=False, preperm=True, log=print):
    import torch
    import torch.nn as nn

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_torch_model(kind, Xtr.shape[2], hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)

    Xtr_t = torch.from_numpy(Xtr).to(dev)
    ytr_t = torch.from_numpy(ytr).to(dev)
    Xva_t = torch.from_numpy(Xva).to(dev)
    yva_t = torch.from_numpy(yva).to(dev)
    Xte_t = torch.from_numpy(Xte).to(dev)
    g = torch.Generator(device=dev if preperm and dev == "cuda" else "cpu").manual_seed(seed)
    lossf = nn.MSELoss()

    def predict(X):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16 and dev == "cuda"):
                    out.append(model(X[i:i + 8192]).float())
        return torch.cat(out)

    N = len(Xtr_t)
    best, bad, best_state, ep_done = float("inf"), 0, None, 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        if preperm and dev == "cuda":
            perm = torch.randperm(N, generator=g, device=dev)
            Xs, ys = Xtr_t[perm], ytr_t[perm]
            for i in range(0, N, batch):
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    loss = lossf(model(Xs[i:i + batch]).float(), ys[i:i + batch])
                loss.backward()
                opt.step()
        else:
            perm = torch.randperm(N, generator=g).to(dev)
            for i in range(0, N, batch):
                b = perm[i:i + batch]
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16 and dev == "cuda"):
                    loss = lossf(model(Xtr_t[b]).float(), ytr_t[b])
                loss.backward()
                opt.step()
        vl = float(lossf(predict(Xva_t), yva_t))
        sched.step(vl)
        ep_done = ep + 1
        if np.isfinite(vl) and vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    pva = predict(Xva_t).cpu().numpy()
    pte = predict(Xte_t).cpu().numpy()
    dt = time.time() - t0
    log(f"    {kind:8s} s={seed} ep={ep_done:2d} val_mse={best:.5f} bf16={use_bf16} t={dt:.1f}s")
    import gc
    del model, opt, sched, Xtr_t, ytr_t, Xva_t, yva_t, Xte_t, predict
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    return pva, pte, best_state, ep_done


def nnls_stack(val_preds, yva, test_preds):
    from scipy.optimize import nnls
    names = list(val_preds)
    A = np.stack([val_preds[n] for n in names], axis=1).astype(np.float64)
    w, _ = nnls(A, yva.astype(np.float64))
    T = np.stack([test_preds[n] for n in names], axis=1).astype(np.float64)
    return dict(zip(names, w)), A @ w, T @ w
