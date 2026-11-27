"""Models. Targets are scaled by (mu, sd); predictions are returned in scaled units."""
import time

import numpy as np

import config


def fit_linear(Xtr, ytr, Xva, Xte, ridge=False):
    from sklearn.linear_model import LinearRegression, Ridge
    m = Ridge(alpha=3.2) if ridge else LinearRegression()
    m.fit(Xtr, ytr)
    return m.predict(Xva), m.predict(Xte)


def fit_gbm(Xtr, ytr, Xva, yva, Xte, seed=0):
    """LightGBM if installed, else sklearn HistGradientBoosting. Returns (pva, pte, name)."""
    try:
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.05, num_leaves=63,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              random_state=seed, verbose=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        return m.predict(Xva), m.predict(Xte), "lightgbm"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed)
        m.fit(Xtr, ytr)
        return m.predict(Xva), m.predict(Xte), "sklearn_hgb (lightgbm not installed)"


def fit_xgb(Xtr, ytr, Xva, yva, Xte, seed=0):
    try:
        import xgboost as xgb
    except ImportError:
        return None
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        dev = "cpu"
    m = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=7, subsample=0.8,
                         colsample_bytree=0.8, early_stopping_rounds=50, random_state=seed,
                         device=dev, tree_method="hist")
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m.predict(Xva), m.predict(Xte)


# ---------------- neural (requires torch; NOT tested in the authoring sandbox) ----------------

def torch_available():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _build(kind, c_in, hidden):
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
            if self.bi:  # forward final state + backward final state (which sits at position 0)
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

    if kind == "gru":
        return RNN("gru", False)
    if kind == "lstm":
        return RNN("lstm", False)
    if kind == "bilstm":
        return RNN("lstm", True)
    if kind == "tcn":
        return TCN()
    raise ValueError(kind)


def train_neural(kind, Xtr, ytr, Xva, yva, Xte, seed, log=print):
    import torch
    import torch.nn as nn

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)
    if dev == "cuda":
        torch.cuda.manual_seed_all(seed)
    use_amp = bool(config.AMP and dev == "cuda")
    model = _build(kind, Xtr.shape[2], config.HIDDEN).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=config.LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    Xtr_t, ytr_t = torch.from_numpy(Xtr).to(dev), torch.from_numpy(ytr).to(dev)
    Xva_t, yva_t = torch.from_numpy(Xva).to(dev), torch.from_numpy(yva).to(dev)
    Xte_t = torch.from_numpy(Xte).to(dev)
    g = torch.Generator().manual_seed(seed)
    lossf = nn.MSELoss()

    def predict(X):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                with torch.autocast("cuda", enabled=use_amp):
                    out.append(model(X[i:i + 8192]).float())
        return torch.cat(out)

    best, bad, best_state, ep_done = float("inf"), 0, None, 0
    t0 = time.time()
    for ep in range(config.EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=g).to(dev)
        for i in range(0, len(perm), config.BATCH):
            b = perm[i:i + config.BATCH]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                loss = lossf(model(Xtr_t[b]).float(), ytr_t[b])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        vl = float(lossf(predict(Xva_t), yva_t))
        sched.step(vl)
        ep_done = ep + 1
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= config.PATIENCE:
                break
    model.load_state_dict(best_state)
    log(f"  {kind} seed={seed} epochs={ep_done} val_mse(scaled)={best:.5f} "
        f"amp_active={use_amp} device={dev} time={time.time() - t0:.0f}s")
    return predict(Xva_t).cpu().numpy(), predict(Xte_t).cpu().numpy()


def nnls_stack(val_preds, yva, test_preds):
    """Non-negative least squares, weights fit on validation only."""
    from scipy.optimize import nnls
    names = list(val_preds)
    A = np.stack([val_preds[n] for n in names], axis=1)
    w, _ = nnls(A.astype(np.float64), yva.astype(np.float64))
    T = np.stack([test_preds[n] for n in names], axis=1)
    return dict(zip(names, w)), A @ w, T @ w
