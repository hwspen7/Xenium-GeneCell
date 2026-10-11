"""Synthetic plot regression tests. Does not open Xenium data or checkpoints.

Run from project root: python -m scripts.test_correlation_plots
"""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.evaluate_sgmae_conditions import (
    SAMPLES,
    _plot_correlations,
    _plot_gene_distribution,
    _plot_scatter,
)


def main():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        synthetic_rows = []
        samples = [x[0] for x in SAMPLES]
        methods = ["Gene-frequency baseline", "baseline", "scheduler"]
        for sample, display, condition, scope in SAMPLES:
            (root / sample).mkdir()
            for method_idx, method in enumerate(methods):
                baseline = method_idx == 0
                synthetic_rows.append({
                    "sample": sample,
                    "method": method,
                    "positive_macro_pearson": None if baseline else 0.45,
                    "positive_macro_spearman": None if baseline else 0.43,
                    "full_macro_pearson": None if baseline else 0.55,
                    "full_macro_spearman": None if baseline else 0.51,
                    "full_mae": 0.45 if baseline else 0.32,
                    "full_pearson": 0.25 if baseline else 0.65,
                })
            for label in ("baseline", "scheduler"):
                rows = []
                for i in range(20):
                    rows.append({
                        "model_positive_pearson": i / 30,
                        "model_positive_spearman": i / 40,
                        "model_full_pearson": i / 25,
                        "model_full_spearman": i / 35,
                    })
                csv_path = root / sample / f"{label}_per_gene.csv"
                with csv_path.open("x", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                x = np.arange(200, dtype=np.float64) / 75
                np.savez_compressed(
                    root / sample / f"{label}_scatter_samples.npz",
                    target_log1p=x,
                    predicted_positive_log1p=x + 0.12,
                    predicted_full_log1p=x * 0.8,
                    is_positive=np.arange(200) % 2 == 0,
                )
        _plot_correlations(synthetic_rows, ["baseline", "scheduler"], root / "correlations.png")
        _plot_gene_distribution(root, "scheduler", root / "gene_distribution.png")
        assert _plot_scatter(root, "scheduler", root / "positive_scatter.png", positive_only=True)
        assert _plot_scatter(root, "scheduler", root / "full_scatter.png", positive_only=False)
        for name in ("correlations.png", "gene_distribution.png", "positive_scatter.png", "full_scatter.png"):
            output = root / name
            assert output.exists() and output.stat().st_size > 10000, name
        assert samples == ["shaml", "shamr", "hour12r", "day2l"]
        print("Four-slice synthetic Pearson/Spearman figures: PASS")


if __name__ == "__main__":
    main()
