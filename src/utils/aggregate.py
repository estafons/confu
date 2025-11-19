import os
import pandas as pd
from glob import glob
import re
import hydra

from textwrap import indent

def pretty_print_results(filename, averages, stds):
    width = max(len(k) for k in averages.keys()) + 2
    print(f"\n  File: {filename}")
    print("  " + "-" * (width + 20))
    print(f"  {'Metric':<{width}}   Mean      Std")
    print("  " + "-" * (width + 20))

    for col in sorted(averages.keys()):
        mean_val = averages[col]
        std_val = stds[col]
        print(f"  {col:<{width}} {mean_val:8.4f} {std_val:8.4f}")


@hydra.main(config_path="../../configs", config_name="config")
def aggregate_multibench(cfg):
    csv_files = glob(os.path.join(cfg.results_path, "*.csv"))

    if not csv_files:
        print("No CSV files found!")
        return

    dataset_groups = {"humor": [], "sarcasm": [], "mosi": []}

    for f in csv_files:
        name = os.path.basename(f).lower()
        if "humor" in name:
            dataset_groups["humor"].append(f)
        elif "sarcasm" in name:
            dataset_groups["sarcasm"].append(f)
        elif "mosi" in name:
            dataset_groups["mosi"].append(f)
        else:
            print(f"⚠️  WARNING: Unmatched to dataset: {f}")

    for dataset, files in dataset_groups.items():
        if not files:
            continue

        print("\n=================================================")
        print(f"                DATASET: {dataset.upper()}")
        print("=================================================\n")

        for csv_file in files:
            filename = os.path.basename(csv_file)
            fn_lower = filename.lower()

            skip_pairwise = any(k in fn_lower for k in ["symile", "triangle"])
            skip_one_to_two = "triclip" in fn_lower

            df = pd.read_csv(csv_file)

            if "AllModalities" in df.columns:
                df = df.rename(columns={"AllModalities": "classification"})

            recall_cols = [col for col in df.columns if "recall@" in col]

            if skip_pairwise:
                recall_cols = [
                    col for col in recall_cols
                    if not re.match(r"^M[123]->_M[123]_recall", col)
                ]

            if skip_one_to_two:
                recall_cols = [
                    col for col in recall_cols
                    if not re.match(r"^(M[123]->_M(12|13|23)|M(12|13|23)->_M[123])_recall", col)
                ]

            if "classification" in df.columns:
                recall_cols.append("classification")

            if not recall_cols:
                print(f"  ⚠️ No recall metrics found after filtering for {filename}")
                continue

            averages = df[recall_cols].mean()
            stds = df[recall_cols].std().fillna(0)

            pretty_print_results(filename, averages, stds)





if __name__ == "__main__":
    aggregate_multibench()