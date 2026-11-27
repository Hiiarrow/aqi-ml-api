"""
End-to-end automated orchestrator (no API keys needed).

Steps executed:
  1. Fetch 6-city CPCB 15-min data (2017-2023) via sparse HTTP Range Parquet reads -> raw_cpcb/
  2. Audit raw station spans, cap/extreme values, and distributions -> fetch/cpcb_station_audit_2017_2023.csv
  3. Export clean hourly OpenCity-format CSVs for the 24 stations -> data/<City>/<Station>.csv
  4. Run pipeline: step0 -> step1 -> step2 -> step3 (5 seeds) -> step4 (bootstrap) -> step5 (regimes)

Stops after the requested fold/horizon (default: Delhi h=1) and prints paths to all outputs.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print(f"\n>>> {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2017)
    ap.add_argument("--end", type=int, default=2023)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--heldout", default="Delhi")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--stations", default=None)
    args = ap.parse_args()

    st_flag = f'--stations "{args.stations}"' if args.stations else ""

    run(f"python -u fetch/download_cpcb.py --start {args.start} --end {args.end}")
    run("python -u fetch/audit_cpcb.py")
    run(f"python -u fetch/export_opencity.py --out-dir {args.data} {st_flag}".strip())

    run(f"python -u run.py step0 --data {args.data} --out {args.out}")
    run(f"python -u run.py step1 --data {args.data} --out {args.out}")
    run(f"python -u run.py step2 --data {args.data} --out {args.out}")
    run(f"python -u run.py step3 --data {args.data} --out {args.out} --heldout {args.heldout} --horizon {args.horizon}")
    run(f"python -u run.py step4 --data {args.data} --out {args.out} --heldout {args.heldout} --horizon {args.horizon}")
    run(f"python -u run.py step5 --data {args.data} --out {args.out} --heldout {args.heldout} --horizon {args.horizon}")

    print(f"\nCompleted automated run for {args.heldout} h={args.horizon}. Outputs saved under {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
