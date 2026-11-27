# `spm_pipeline` — Core Pipeline & Final Study Runner

See the repository root [README](../README.md) for the complete study overview, methodology, and benchmark tables.

## Main Entrypoints
- `final_study.py` — Runs the 20-station frozen QC v3 suite (24 runs, all 5 neural families × 5 seeds, baselines, DATE-block paired bootstrap CIs, and time-t regime breakdowns) into `results_final/`.
- `final_report.py` — Aggregates `results_final/` CSVs into `FINAL_REPORT.md` and `TABLE1..TABLE4` CSVs.
- `make_readme_figures.py` — Renders neutral-palette figures into `../assets/`.
- `fetch/ingest_2024.py` — Downloads/caches 2024 CPCB + Open-Meteo ERA5 data and builds frozen QC v3 caches.
- `qc.py` — `qc_station` (v1), `qc_station_v2` (v2), and frozen `qc_station_v3` (v3).

## Verification
```bash
python tests/test_pipeline.py
python tests/test_sweep.py
python tests/test_final.py
```
