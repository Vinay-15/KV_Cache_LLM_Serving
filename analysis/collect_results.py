#!/usr/bin/env python3
import glob
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "results", "all_results.csv")

def main():
    rows = []

    for path in glob.glob(os.path.join(RESULTS, "*", "request_summary.csv")):
        df = pd.read_csv(path)
        rows.append(df)

    if not rows:
        raise SystemExit("No request_summary.csv files found.")

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["workload", "concurrency"])
    out.to_csv(OUT, index=False)

    print(out.to_string(index=False))
    print(f"\nWrote {OUT}")

if __name__ == "__main__":
    main()

