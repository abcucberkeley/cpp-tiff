#!/usr/bin/env python3
"""Compare two labeled benchmark runs from benchResults.csv.

Usage:
    python3 benchmarks/compare.py <baseline_label> <new_label> [csv_path]

For each file benchmarked under both labels it prints the median read time and
the speedup (baseline / new). If a label was run more than once for a file, the
most recent row in the CSV is used. Only the Python standard library is used.
"""
import csv
import sys
from collections import OrderedDict


def load(path):
    # label -> (file -> row); later rows overwrite earlier ones (latest wins).
    data = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            data.setdefault(r["label"], OrderedDict())[r["file"]] = r
    return data


def fmt(x, width=10, prec=2):
    return f"{x:>{width}.{prec}f}"


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    base_label, new_label = sys.argv[1], sys.argv[2]
    path = sys.argv[3] if len(sys.argv) > 3 else "benchResults.csv"

    data = load(path)
    for lbl in (base_label, new_label):
        if lbl not in data:
            print(f"Label '{lbl}' not found in {path}. Available: {', '.join(data)}")
            sys.exit(1)

    base, new = data[base_label], data[new_label]
    common = [f for f in base if f in new]
    if not common:
        print("No files in common between the two labels.")
        sys.exit(1)
    # Order by logical volume size ascending for a readable progression.
    common.sort(key=lambda f: int(base[f]["logical_bytes"]))

    print(f"Comparing  baseline='{base_label}'  vs  new='{new_label}'  ({path})\n")
    print(f"{'file':<34} {'size':>8} {'base_ms':>11} {'new_ms':>11} "
          f"{'speedup':>9} {'base_GB/s':>10} {'new_GB/s':>10}")
    print("-" * 98)

    speedups = []
    for f in common:
        b, n = base[f], new[f]
        bms, nms = float(b["median_ms"]), float(n["median_ms"])
        speedup = bms / nms if nms > 0 else float("nan")
        speedups.append(speedup)
        gb = float(b["logical_bytes"]) / 1e9
        disp = f if len(f) <= 34 else "..." + f[-(34 - 3):]
        print(f"{disp:<34} {gb:>7.2f}G {fmt(bms, 11)} {fmt(nms, 11)} "
              f"{speedup:>8.3f}x {float(b['gbps_median']):>10.2f} "
              f"{float(n['gbps_median']):>10.2f}")

    print("-" * 98)
    if speedups:
        geo = 1.0
        for s in speedups:
            geo *= s
        geo **= 1.0 / len(speedups)
        print(f"geomean speedup: {geo:.3f}x  (>1 means '{new_label}' is faster)")


if __name__ == "__main__":
    main()
