"""
Generate clean neutral-palette summary figures in ../assets/ strictly from
verified results_final/ CSVs for the repository README.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL_DIR = os.path.join(HERE, "results_final")
ASSETS_DIR = os.path.join(os.path.dirname(HERE), "assets")

# Neutral editorial palette (charcoal / slate / warm stone / muted ash)
C_INK = "#111827"
C_CHARCOAL = "#1F2937"
C_SLATE_DARK = "#334155"
C_SLATE_MID = "#475569"
C_TAUPE = "#78716C"
C_ASH = "#9CA3AF"
C_STONE = "#D6D3D1"
C_GRID = "#E5E7EB"
C_BG = "#FFFFFF"
C_PANEL = "#FAFAFA"

plt.rcParams.update({
    "figure.facecolor": C_BG,
    "axes.facecolor": C_PANEL,
    "axes.edgecolor": "#D1D5DB",
    "axes.linewidth": 0.8,
    "axes.labelcolor": C_INK,
    "axes.titlecolor": C_INK,
    "xtick.color": "#374151",
    "ytick.color": "#374151",
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
})


def make_fig1_temporal_horizons(t1):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), gridspec_kw={"width_ratios": [1.55, 1.0]}, sharey=True)

    specs = [
        (axes[0], "P2_2023_ALL20", [1, 6, 12, 24], ["h = 1", "h = 6", "h = 12", "h = 24"], "Temporal Holdout 2023 (Train 2017-2022, 20 Stations)"),
        (axes[1], "P2_2024_ALL19", [1, 24], ["h = 1", "h = 24"], "Temporal Holdout 2024 (Train 2017-2023, 19 Stations)"),
    ]

    bar_specs = [
        ("ridge (none)", C_STONE),
        ("best_tree (none)", C_ASH),
        ("equal_neural_ens (none, Primary)", C_CHARCOAL),
        ("equal_neural_ens (+Open-Meteo)", C_SLATE_MID),
    ]
    w = 0.19

    for ax, prefix, hors, xlabels, title in specs:
        x = np.arange(len(hors))
        vals_by_bar = {lbl: [] for lbl, _ in bar_specs}
        for h in hors:
            rn = t1[t1.run_id == f"{prefix}_h{h}_direct_none"].iloc[0]
            ro = t1[t1.run_id == f"{prefix}_h{h}_direct_openmeteo"].iloc[0]
            vals_by_bar["ridge (none)"].append(float(rn["ridge_skill"]))
            vals_by_bar["best_tree (none)"].append(float(rn["best_tree_skill_RMSE"]))
            vals_by_bar["equal_neural_ens (none, Primary)"].append(float(rn["equal_neural_ens_skill_RMSE"]))
            vals_by_bar["equal_neural_ens (+Open-Meteo)"].append(float(ro["equal_neural_ens_skill_RMSE"]))

        offsets = np.linspace(-1.5 * w, 1.5 * w, len(bar_specs))
        for off, (lbl, col) in zip(offsets, bar_specs):
            ys = vals_by_bar[lbl]
            bars = ax.bar(x + off, ys, width=w * 0.92, color=col, edgecolor="#FFFFFF", linewidth=0.5, label=lbl, zorder=3)
            for b, v in zip(bars, ys):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=7.6, color=C_INK)

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=10)
        ax.grid(axis="y", color=C_GRID, linestyle="--", linewidth=0.7, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, 42)

    axes[0].set_ylabel("RMSE Skill vs Persistence (%)", fontsize=10, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False, fontsize=8.8)
    fig.suptitle("Temporal Holdout Skill Across Forecast Horizons (QC v3, 5 Neural Families x 5 Seeds)", fontsize=11.8, fontweight="bold", y=1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(ASSETS_DIR, "fig1_temporal_horizons.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


def make_fig2_loco_and_weather_ci(t1, om_ci):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2), gridspec_kw={"width_ratios": [1.05, 1.15]})

    # Left: LOCO unseen-city generalization (direct_none primary vs best_tree vs ridge + direct_openmeteo ens)
    loco_labels = ["Kolkata\n(h=1)", "Delhi\n(h=1)", "Bengaluru\n(h=1)", "Chennai\n(h=1)", "Mumbai\n(h=1)", "Delhi\n(h=24)"]
    loco_keys = [("Kolkata", 1), ("Delhi", 1), ("Bengaluru", 1), ("Chennai", 1), ("Mumbai", 1), ("Delhi", 24)]
    y = np.arange(len(loco_keys))[::-1]
    h_bar = 0.22

    ridge_v, tree_v, ens_n_v, ens_o_v = [], [], [], []
    for city, hor in loco_keys:
        rn = t1[t1.run_id == f"P1_LOCO_{city}_h{hor}_direct_none"].iloc[0]
        ro = t1[t1.run_id == f"P1_LOCO_{city}_h{hor}_direct_openmeteo"].iloc[0]
        ridge_v.append(float(rn["ridge_skill"]))
        tree_v.append(float(rn["best_tree_skill_RMSE"]))
        ens_n_v.append(float(rn["equal_neural_ens_skill_RMSE"]))
        ens_o_v.append(float(ro["equal_neural_ens_skill_RMSE"]))

    ax1.barh(y + 1.5 * h_bar, ens_n_v, height=h_bar * 0.9, color=C_CHARCOAL, label="equal_neural_ens (none, Primary)", zorder=3)
    ax1.barh(y + 0.5 * h_bar, tree_v, height=h_bar * 0.9, color=C_ASH, label="best_tree (val-selected residual/direct)", zorder=3)
    ax1.barh(y - 0.5 * h_bar, ridge_v, height=h_bar * 0.9, color=C_STONE, label="ridge (none)", zorder=3)
    ax1.barh(y - 1.5 * h_bar, ens_o_v, height=h_bar * 0.9, color=C_TAUPE, alpha=0.85, label="equal_neural_ens (+Open-Meteo)", zorder=3)

    for yi, v_n, v_o in zip(y, ens_n_v, ens_o_v):
        ax1.text(max(v_n, 0) + 0.35, yi + 1.5 * h_bar, f"{v_n:+.1f}%", va="center", ha="left", fontsize=7.6, color=C_INK, fontweight="bold")
        if v_o >= 0:
            ax1.text(v_o + 0.35, yi - 1.5 * h_bar, f"{v_o:+.1f}%", va="center", ha="left", fontsize=7.4, color=C_TAUPE)
        else:
            ax1.text(0.45, yi - 1.5 * h_bar, f"{v_o:+.1f}% (911 hPa OOD)", va="center", ha="left", fontsize=7.4, color=C_TAUPE, fontweight="bold")

    ax1.axvline(0, color="#6B7280", linewidth=0.9, zorder=2)
    ax1.set_yticks(y)
    ax1.set_yticklabels(loco_labels, fontsize=8.8)
    ax1.set_xlim(-26.5, 18.0)
    ax1.set_xlabel("RMSE Skill vs Persistence (%)", fontsize=9.5, fontweight="bold")
    ax1.set_title("Leave-One-City-Out (LOCO) Zero-Shot Skill", fontsize=10.5, fontweight="bold", pad=10)
    ax1.grid(axis="x", color=C_GRID, linestyle="--", linewidth=0.7, zorder=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.45, -0.15), ncol=2, fontsize=7.6, frameon=False)

    # Right: Forest plot of ALL 12 Open-Meteo vs None DATE-block 95% CIs (equal_neural_ens[om] - [none])
    ci_order = [
        ("P2_2023", "ALL20", 1,  "P2 2023 (20 stn) h=1"),
        ("P2_2023", "ALL20", 6,  "P2 2023 (20 stn) h=6"),
        ("P2_2023", "ALL20", 12, "P2 2023 (20 stn) h=12"),
        ("P2_2023", "ALL20", 24, "P2 2023 (20 stn) h=24"),
        ("P2_2024", "ALL19", 1,  "P2 2024 (19 stn) h=1"),
        ("P2_2024", "ALL19", 24, "P2 2024 (19 stn) h=24"),
        ("P1_LOCO", "Delhi", 24, "LOCO Delhi h=24"),
        ("P1_LOCO", "Mumbai", 1, "LOCO Mumbai h=1"),
        ("P1_LOCO", "Kolkata", 1,"LOCO Kolkata h=1"),
        ("P1_LOCO", "Chennai", 1,"LOCO Chennai h=1"),
        ("P1_LOCO", "Delhi", 1,  "LOCO Delhi h=1"),
        ("P1_LOCO", "Bengaluru", 1, "LOCO Bengaluru h=1"),
    ]
    yc = np.arange(len(ci_order))[::-1]
    lbls_c, means, los, his = [], [], [], []
    for p, hld, hor, lbl in ci_order:
        r = om_ci[(om_ci.protocol == p) & (om_ci.heldout == hld) & (om_ci.horizon == hor)].iloc[0]
        lbls_c.append(lbl)
        means.append(float(r["dRMSE_mean"]))
        los.append(float(r["dRMSE_lo"]))
        his.append(float(r["dRMSE_hi"]))

    for yi, m, lo, hi in zip(yc, means, los, his):
        col = C_CHARCOAL if hi < 0 else (C_TAUPE if lo > 0 else C_ASH)
        ax2.plot([lo, hi], [yi, yi], color=col, linewidth=2.0, zorder=3)
        ax2.scatter([m], [yi], color=col, s=36, zorder=4)
        txt = f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"
        if hi < -0.5 or m > 2.5:
            ax2.text(lo - 0.18, yi, txt, va="center", ha="right", fontsize=7.4, color=col, fontweight="bold" if (hi < -0.5 or m > 2.5) else "normal")
        else:
            ax2.text(max(hi, 0.0) + 0.18, yi, txt, va="center", ha="left", fontsize=7.4, color=col)

    ax2.axvline(0, color="#6B7280", linestyle="--", linewidth=1.0, zorder=2)
    ax2.axhline(5.5, color="#D1D5DB", linestyle=":", linewidth=1.0, zorder=1)
    ax2.set_yticks(yc)
    ax2.set_yticklabels(lbls_c, fontsize=8.4)
    ax2.set_xlim(-5.2, 5.4)
    ax2.set_xlabel("dRMSE (Open-Meteo minus None, ug/m3; negative = helps)", fontsize=9.2, fontweight="bold")
    ax2.set_title("Paired DATE-Block 95% CIs: Adding Lagged Open-Meteo", fontsize=10.5, fontweight="bold", pad=10)
    ax2.grid(axis="x", color=C_GRID, linestyle="--", linewidth=0.7, zorder=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Spatial Generalization (LOCO) and Lagged Weather Ablation (DATE-Block 95% CIs)", fontsize=11.8, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(ASSETS_DIR, "fig2_loco_and_weather_ci.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


def make_fig3_regime_heatmap(reg):
    runs = [
        ("P2_2023_ALL20_h1_direct_none",    "P2 2023 h=1 (none)"),
        ("P2_2023_ALL20_h6_direct_none",    "P2 2023 h=6 (none)"),
        ("P2_2023_ALL20_h12_direct_none",   "P2 2023 h=12 (none)"),
        ("P2_2023_ALL20_h24_direct_none",   "P2 2023 h=24 (none)"),
        ("P2_2024_ALL19_h1_direct_none",    "P2 2024 h=1 (none)"),
        ("P2_2024_ALL19_h24_direct_none",   "P2 2024 h=24 (none)"),
        ("P1_LOCO_Delhi_h1_direct_none",    "LOCO Delhi h=1 (none)"),
        ("P1_LOCO_Delhi_h24_direct_none",   "LOCO Delhi h=24 (none)"),
        ("P1_LOCO_Mumbai_h1_direct_none",   "LOCO Mumbai h=1 (none)"),
        ("P1_LOCO_Kolkata_h1_direct_none",  "LOCO Kolkata h=1 (none)"),
        ("P1_LOCO_Chennai_h1_direct_none",  "LOCO Chennai h=1 (none)"),
        ("P1_LOCO_Bengaluru_h1_direct_none","LOCO Bengaluru h=1 (none)"),
    ]
    reg_cols = [
        ("y_t >= 60", "y_t >= 60"),
        ("y_t >= 120", "y_t >= 120"),
        ("y_t >= 250", "y_t >= 250"),
        ("onset (y_t - y_t-3 > 30)", "Onset (+30/3h)"),
        ("winter (Nov-Feb)", "Winter (Nov-Feb)"),
        ("other months (Mar-Oct)", "Other (Mar-Oct)"),
        ("all", "Overall (all)"),
    ]

    sub = reg[reg.model == "equal_neural_ens"]
    mat_r = np.zeros((len(runs), len(reg_cols)))
    mat_m = np.zeros((len(runs), len(reg_cols)))
    for i, (rid, _) in enumerate(runs):
        sr = sub[sub.run_id == rid].set_index("regime")
        for j, (rc, _) in enumerate(reg_cols):
            mat_r[i, j] = float(sr.loc[rc, "skill_RMSE_pct"])
            mat_m[i, j] = float(sr.loc[rc, "skill_MAE_pct"])

    cmap = LinearSegmentedColormap.from_list("neutral_slate", ["#F4F4F5", "#D4D4D8", "#94A3B8", "#475569", "#1E293B"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 6.2), sharey=True)

    for ax, mat, title in [
        (ax1, mat_r, "Primary Ensemble (direct_none): RMSE Skill (%) by Regime"),
        (ax2, mat_m, "Primary Ensemble (direct_none): MAE Skill (%) by Regime"),
    ]:
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=-5, vmax=50)
        ax.set_xticks(np.arange(len(reg_cols)))
        ax.set_xticklabels([lbl for _, lbl in reg_cols], rotation=24, ha="right", fontsize=8.4)
        ax.set_title(title, fontsize=10.2, fontweight="bold", pad=10)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                txt_col = "#FFFFFF" if v >= 24 else C_INK
                ax.text(j, i, f"{v:+.1f}%", ha="center", va="center", fontsize=7.9, color=txt_col, fontweight="medium")
        for xline in np.arange(-0.5, len(reg_cols), 1):
            ax.axvline(xline, color="#FFFFFF", linewidth=1.2)
        for yline in np.arange(-0.5, len(runs), 1):
            ax.axhline(yline, color="#FFFFFF", linewidth=1.2)

    ax1.set_yticks(np.arange(len(runs)))
    ax1.set_yticklabels([lbl for _, lbl in runs], fontsize=8.8)
    ax1.axhline(5.5, color=C_CHARCOAL, linewidth=1.8)
    ax2.axhline(5.5, color=C_CHARCOAL, linewidth=1.8)
    cbar = fig.colorbar(im, ax=[ax1, ax2], fraction=0.018, pad=0.025)
    cbar.set_label("Skill vs Persistence (%)", fontsize=9, fontweight="bold")
    fig.suptitle("Primary Fixed Method (equal_neural_ens, direct_none) Across Pollution & Seasonal Regimes", fontsize=11.6, fontweight="bold", y=0.99)
    out = os.path.join(ASSETS_DIR, "fig3_regime_skill_heatmap.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    t1 = pd.read_csv(os.path.join(FINAL_DIR, "TABLE1_BASELINES_AND_MODELS.csv"))
    om_ci = pd.read_csv(os.path.join(FINAL_DIR, "OPENMETEO_VS_NONE_CIS.csv"))
    reg = pd.read_csv(os.path.join(FINAL_DIR, "TABLE4_REGIMES_CLEAN.csv"))

    make_fig1_temporal_horizons(t1)
    make_fig2_loco_and_weather_ci(t1, om_ci)
    make_fig3_regime_heatmap(reg)


if __name__ == "__main__":
    main()
