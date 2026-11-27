"""Makes FAKE OpenCity-format CSVs to test the code. Not real data; never use for results."""
import calendar
import os
import sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "synth_data"
rng = np.random.default_rng(1)
cities = {"Delhi": (190, ["Anand Vihar", "Mandir Marg"]), "Mumbai": (110, ["BKC", "Powai", "Sion MPCB"]),
          "Kolkata": (100, ["Ballygunge", "Jadavpur"]), "Chennai": (70, ["Manali", "Royapuram"]),
          "Bengaluru": (65, ["BTM", "Peenya"]), "Hyderabad": (90, ["Zoo Park", "Sanathnagar"])}


def series(level, y0, y1):
    n = int((np.datetime64(f"{y1}-12-31T23") - np.datetime64(f"{y0}-01-01T00")) / np.timedelta64(1, "h")) + 1
    t = np.arange(n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.93 * x[i - 1] + rng.normal(0, 0.18)
    season = 0.5 * np.cos(2 * np.pi * (t / 8766 - 0.05))
    diurnal = 0.2 * np.cos(2 * np.pi * (t % 24 - 9) / 24)
    return level * np.exp(x + season + diurnal - 0.2)


def write(path, v, y0, y1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    k = 0
    with open(path, "w") as f:
        for y in range(y0, y1 + 1):
            f.write(f"Year,{y}\n")
            for m in range(1, 13):
                f.write(f"{calendar.month_name[m]}-{y}," + ",".join(f"{h:02d}:00:00" for h in range(24)) + "\n")
                if m == 1 and y == y0 and k == 0:
                    f.write('"\n"\n')
                for d in range(1, calendar.monthrange(y, m)[1] + 1):
                    row = v[k:k + 24]
                    k += 24
                    f.write(f"{d}," + ",".join("" if np.isnan(x) else f"{x:.1f}" for x in row) + "\n")


for city, (level, stations) in cities.items():
    for j, s in enumerate(stations):
        y0, y1 = 2017, 2019
        v = np.minimum(series(level * (1 + 0.1 * j), y0, y1), 500.0)
        n = len(v)
        for _ in range(40):  # random gaps
            a = rng.integers(0, n - 30)
            v[a:a + rng.integers(1, 12)] = np.nan
        for _ in range(5):  # stuck sensor
            a = rng.integers(0, n - 30)
            v[a:a + 10] = round(float(np.nanmedian(v)), 1)
        if city == "Kolkata" and j == 0:
            v[5000] = 3562.0
        write(os.path.join(root, city, s + ".csv"), np.round(v, 1), y0, y1)
# a short-span station to test the span rule
v = series(80, 2019, 2019)[:5000]
write(os.path.join(root, "Mumbai", "Chembur.csv"), np.pad(v, (0, 8760 - 5000), constant_values=np.nan), 2019, 2019)
print("synthetic data written to", root)
