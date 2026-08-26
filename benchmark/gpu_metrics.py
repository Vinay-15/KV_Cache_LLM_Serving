#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import subprocess
import time

FIELDS = [
    "timestamp",
    "index",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "temperature.gpu",
]

QUERY = ",".join(FIELDS[1:])

def sample():
    cmd = [
        "nvidia-smi",
        f"--query-gpu={QUERY}",
        "--format=csv,noheader,nounits",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    rows = []
    for line in result.stdout.strip().splitlines():
        values = [x.strip() for x in line.split(",")]
        rows.append(values)

    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "run_id"] + FIELDS[1:])

        print(f">> GPU telemetry -> {args.csv}")

        try:
            while True:
                timestamp = dt.datetime.now(dt.timezone.utc).isoformat()

                try:
                    rows = sample()
                    for values in rows:
                        writer.writerow(
                            [timestamp, args.run_id] + values
                        )
                    f.flush()
                except Exception as e:
                    print(f"   nvidia-smi failed: {e}")

                time.sleep(args.interval)

        except KeyboardInterrupt:
            print("\n>> GPU collector stopped")

if __name__ == "__main__":
    main()

