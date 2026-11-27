"""
Automated zero-key CPCB 15-minute data fetcher for 6 Indian cities (2017-2023+).
Reads official CPCB archive Parquets from Vonter/india-cpcb-aqi GitHub Releases
using exact Parquet footer inspection + sparse HTTP Range requests (transferring
only matching State/City row groups and selected columns).

Outputs per-year cached Parquets:
    raw_cpcb/cpcb_6cities_<year>.parquet
"""
import argparse
import io
import os
import struct
import time
from bisect import bisect_right

import pyarrow.parquet as pq
import requests

TARGET_STATES = {"Delhi", "Karnataka", "Maharashtra", "Tamil Nadu", "Telangana", "West Bengal"}
TARGET_CITIES = {"Delhi", "Bengaluru", "Mumbai", "Chennai", "Hyderabad", "Kolkata"}
KEEP_COLS = [
    "Station ID", "State", "City", "Station Name", "Timestamp",
    "PM2.5 (µg/m³)", "PM10 (µg/m³)",
    "AT (°C)", "RH (%)", "WS (m/s)", "WD (deg)", "RF (mm)", "SR (W/mt2)", "BP (mmHg)",
]


def http_get_range(sess, url, start, end, retries=5):
    for attempt in range(retries):
        try:
            r = sess.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=90)
            r.raise_for_status()
            return r.content
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"    retry {attempt+1}/{retries} range {start}-{end} after {wait}s: {e}", flush=True)
            time.sleep(wait)


def fetch_year_sparse(yr, out_path):
    t0 = time.time()
    url = f"https://github.com/Vonter/india-cpcb-aqi/releases/download/{yr}/cpcb-air-quality-{yr}.parquet"
    sess = requests.Session()
    sess.headers["User-Agent"] = "python-cpcb-fetcher"

    # Resolve redirect + get exact total Content-Length via 1-byte range request
    r0 = sess.get(url, headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=30)
    r0.raise_for_status()
    final_url = r0.url
    cr = r0.headers.get("Content-Range", "")
    if "/" not in cr:
        raise RuntimeError(f"Server did not return Content-Range for {url}")
    total_len = int(cr.split("/")[-1])

    # Read exact Parquet footer: last 8 bytes = <uint32 footer_len><b'PAR1'>
    tail8 = http_get_range(sess, final_url, total_len - 8, total_len - 1)
    footer_len, magic = struct.unpack("<I4s", tail8)
    if magic != b"PAR1":
        raise RuntimeError(f"Invalid Parquet magic {magic} for {yr}")

    meta_start = total_len - 8 - footer_len
    meta_bytes = http_get_range(sess, final_url, meta_start, total_len - 1)

    class MetaOnlyFile(io.RawIOBase):
        def __init__(self):
            self.pos = 0
        def readable(self): return True
        def seekable(self): return True
        def tell(self): return self.pos
        def seek(self, off, w=io.SEEK_SET):
            if w == io.SEEK_SET: self.pos = off
            elif w == io.SEEK_CUR: self.pos += off
            elif w == io.SEEK_END: self.pos = total_len + off
            return self.pos
        def read(self, n=-1):
            if n == 0 or self.pos >= total_len: return b""
            if self.pos < meta_start:
                raise RuntimeError(f"MetaOnlyFile unexpected seek before footer: {self.pos}")
            o = self.pos - meta_start
            d = meta_bytes[o:] if n < 0 else meta_bytes[o:o + n]
            self.pos += len(d)
            return d
        def readinto(self, b):
            d = self.read(len(b))
            n = len(d)
            b[:n] = d
            return n

    pf = pq.ParquetFile(MetaOnlyFile())
    names = pf.schema.names
    st_idx = names.index("State")
    ct_idx = names.index("City")
    cols_present = [c for c in KEEP_COLS if c in names]
    col_idxs = [names.index(c) for c in cols_present]

    rgs, spans = [], []
    for i in range(pf.metadata.num_row_groups):
        rg = pf.metadata.row_group(i)
        st_stats = rg.column(st_idx).statistics
        ct_stats = rg.column(ct_idx).statistics
        if st_stats is None or not st_stats.has_min_max:
            st_ok = True
        else:
            smin, smax = st_stats.min, st_stats.max
            st_ok = any(smin <= ts <= smax for ts in TARGET_STATES)
        if not st_ok:
            continue
        if st_stats is not None and st_stats.has_min_max and st_stats.min == st_stats.max:
            if ct_stats is not None and ct_stats.has_min_max:
                cmin, cmax = ct_stats.min, ct_stats.max
                if not any(cmin <= tc <= cmax for tc in TARGET_CITIES):
                    continue
        rgs.append(i)
        for ci in col_idxs:
            cm = rg.column(ci)
            offs = [x for x in (cm.dictionary_page_offset, cm.data_page_offset) if x is not None and x >= 0]
            s = min(offs)
            e = min(total_len - 1, s + cm.total_compressed_size + 128)
            spans.append((s, e))

    spans.sort()
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1] + 524288:  # coalesce gaps <= 512KB
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    dl_mb = (sum(e - s + 1 for s, e in merged) + len(meta_bytes)) / 1e6
    print(f"[{yr}] {pf.metadata.num_rows:,} total rows | rgs={len(rgs)}/{pf.metadata.num_row_groups} | "
          f"fetching {dl_mb:.1f}MB / {total_len/1e6:.1f}MB in {len(merged)} chunks", flush=True)

    chunks = []
    for k, (s, e) in enumerate(merged, 1):
        buf = http_get_range(sess, final_url, s, e)
        chunks.append((s, buf))
        if k % 5 == 0 or k == len(merged):
            print(f"[{yr}]   chunk {k}/{len(merged)} ({time.time()-t0:.1f}s)", flush=True)
    chunks.append((meta_start, meta_bytes))
    chunks.sort(key=lambda x: x[0])
    starts = [c[0] for c in chunks]

    class SparseMemFile(io.RawIOBase):
        def __init__(self):
            self.pos = 0
        def readable(self): return True
        def seekable(self): return True
        def tell(self): return self.pos
        def seek(self, off, w=io.SEEK_SET):
            if w == io.SEEK_SET: self.pos = off
            elif w == io.SEEK_CUR: self.pos += off
            elif w == io.SEEK_END: self.pos = total_len + off
            return self.pos
        def read(self, n=-1):
            if n == 0 or self.pos >= total_len: return b""
            j = bisect_right(starts, self.pos) - 1
            if j >= 0:
                cs, buf = chunks[j]
                if cs <= self.pos < cs + len(buf):
                    o = self.pos - cs
                    d = buf[o:] if n < 0 else buf[o:o + n]
                    self.pos += len(d)
                    return d
            end = min(total_len - 1, self.pos + (1048576 if n < 0 else n) - 1)
            d = http_get_range(sess, final_url, self.pos, end)
            self.pos += len(d)
            return d
        def readinto(self, b):
            d = self.read(len(b))
            n = len(d)
            b[:n] = d
            return n

    pf2 = pq.ParquetFile(SparseMemFile())
    tbl = pf2.read_row_groups(rgs, columns=cols_present)
    df = tbl.to_pandas()
    df = df[df["City"].isin(TARGET_CITIES)].copy()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[{yr}] Saved {len(df):,} rows -> {out_path} ({time.time()-t0:.1f}s)", flush=True)


def extract_local_parquet(local_path, out_path, yr):
    t0 = time.time()
    pf = pq.ParquetFile(local_path)
    cols = [c for c in KEEP_COLS if c in pf.schema.names]
    df = pf.read(columns=cols).to_pandas()
    df = df[df["City"].isin(TARGET_CITIES)].copy()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[{yr}] Extracted locally {len(df):,} rows -> {out_path} ({time.time()-t0:.1f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2017)
    ap.add_argument("--end", type=int, default=2023)
    ap.add_argument("--raw-dir", default="raw_cpcb")
    args = ap.parse_args()

    for yr in range(args.start, args.end + 1):
        out_path = os.path.join(args.raw_dir, f"cpcb_6cities_{yr}.parquet")
        if os.path.exists(out_path):
            print(f"[{yr}] SKIP (already cached at {out_path})", flush=True)
            continue
        full_local = os.path.join(args.raw_dir, f"cpcb-air-quality-{yr}.parquet")
        if os.path.exists(full_local):
            extract_local_parquet(full_local, out_path, yr)
        else:
            fetch_year_sparse(yr, out_path)


if __name__ == "__main__":
    main()
