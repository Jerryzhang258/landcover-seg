"""Aggregate per-run training summaries and test reports into a single CSV."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--csv", default="outputs/results.csv")
    args = ap.parse_args()

    rows = []
    for summary in sorted(glob.glob(os.path.join(args.outputs, "*_summary.json"))):
        with open(summary) as f:
            s = json.load(f)
        row = {
            "run_name": s.get("run_name"),
            "model": s.get("model"),
            "loss": s.get("loss"),
            "params_M": s.get("params_M"),
            "best_val_mIoU": s.get("best_val_mIoU"),
            "train_time_hr": s.get("train_time_hr"),
            "gpu_peak_MB": s.get("gpu_peak_MB"),
        }
        # Try to pair with a test report.
        run_dir = os.path.join(args.outputs, s.get("run_name", ""))
        report_path = os.path.join(run_dir, "report.json")
        if os.path.exists(report_path):
            with open(report_path) as f:
                r = json.load(f)
            row["test_mIoU"] = r.get("mIoU")
            row["test_mDice"] = r.get("mDice")
            row["test_pix_acc"] = r.get("pixel_acc")
            row["infer_ms_per_image"] = r.get("inference_ms_per_image")
            for c, v in r.get("per_class_IoU", {}).items():
                row[f"IoU_{c}"] = v
        rows.append(row)

    if not rows:
        print("[warn] no summaries found in", args.outputs)
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[done] wrote {args.csv}  ({len(rows)} runs)")


if __name__ == "__main__":
    main()
