"""Frozen settings. Edit here once, before running anything. Do not change between folds."""

# POLLUTANT LABEL: verified from official CPCB column "PM2.5 (µg/m³)". Used in all outputs.
POLLUTANT = "PM2.5"

# ---- QC spec (applied once, to every station) ----
CAP = 500.0               # frozen baseline/sweep upper bound (AQI-era cap; proposed <=980 requires retrain)
MAX_STUCK = 6             # runs of identical readings longer than this -> NaN
MAX_GAP = 6               # interpolate gaps up to this many hours only
MIN_SPAN_YEARS = 2.0      # drop stations spanning less than this

# ---- Windowing / split ----
LOOKBACK = 24
HORIZON = 1
VAL_FRACTION = 0.10       # last 10% of each training station's windows

# ---- Training config (same for every model and fold) ----
SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 60
PATIENCE = 7
LR = 1e-3
BATCH = 2048
HIDDEN = 64
AMP = False               # mixed precision off by default; logged if turned on
NEURAL = ["gru", "lstm", "bilstm", "tcn"]

# ---- Evaluation ----
BOOT_RESAMPLES = 1000
BOOT_SEED = 0
SPIKE_ONSET_RISE = 30.0   # y_t - y_{t-3} > this
THRESHOLDS = [60.0, 120.0, 250.0]  # CPCB NAQI PM2.5 24h category boundaries (<=60 Sat., <=120 Poor, <=250 V.Poor, >250 Severe)
WINTER_MONTHS = [11, 12, 1, 2]
