"""
Generate a complete paper-ready data & fold summary from step0/step1/clean.pkl.
Usage: python make_data_summary.py --out results
"""
import argparse
import os
import pickle

import pandas as pd

import config
import data as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    s0 = pd.read_csv(os.path.join(args.out, "step0_files.csv"))
    s1 = pd.read_csv(os.path.join(args.out, "step1_qc_report.csv"))
    dist = pd.read_csv(os.path.join(args.out, "step1_distribution.csv"))
    with open(os.path.join(args.out, "clean.pkl"), "rb") as f:
        clean = pickle.load(f)

    st_tab = (
        s0[["city", "station", "start_date", "end_date", "hourly_slots", "nonnull_raw", "duplicate_ts"]]
        .merge(s1.drop(columns=["nonnull_raw", "duplicate_timestamps"]), on=["city", "station"])
        .merge(dist.drop(columns=["N"]), on=["city", "station"], how="left")
    )
    st_tab.to_csv(os.path.join(args.out, "paper_station_summary.csv"), index=False)

    city_tab = (
        st_tab[st_tab["status"] == "retained"]
        .groupby("city", as_index=False)
        .agg(
            stations=("station", "count"),
            raw_slots=("hourly_slots", "sum"),
            valid_hours_after_qc=("valid_hours_out", "sum"),
            above_cap_removed=("above_cap_removed", "sum"),
            stuck_removed=("stuck_hours_removed", "sum"),
            interpolated=("interpolated_hours", "sum"),
            mean_pm25=("mean", "mean"),
        )
    )
    city_tab.to_csv(os.path.join(args.out, "paper_city_summary.csv"), index=False)

    fold_rows = []
    for heldout in sorted(clean.keys()):
        f = D.build_fold(clean, heldout, config.LOOKBACK, 1)
        fold_rows.append(dict(
            protocol="P1_LOCO",
            heldout_city=heldout,
            horizon=1,
            train_stations=len(set(f["train"]["station"])),
            test_stations=len(set(f["test"]["station"])),
            train_windows=len(f["train"]["y"]),
            val_windows=len(f["val"]["y"]),
            test_windows=len(f["test"]["y"]),
            train_mu=round(f["mu"], 3),
            train_sd=round(f["sd"], 3),
        ))
    ft = pd.DataFrame(fold_rows)
    ft.to_csv(os.path.join(args.out, "paper_loco_fold_sizes.csv"), index=False)

    print("=== City Summary ===")
    print(city_tab.round(2).to_string(index=False))
    print("\n=== LOCO Fold Sizes (h=1, L=24) ===")
    print(ft.to_string(index=False))


if __name__ == "__main__":
    main()
