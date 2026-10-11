"""Regression checks for Pearson, Spearman and complete log-scale reconstruction.

Run from project root: python -m tests.v1.test_correlations
Synthetic data only. No ShamR, Hour12R or Day2L files are loaded.
"""

import math

import numpy as np

from src.evaluation.evaluate_sgmae_v1 import (
    _safe_pearson,
    _safe_spearman,
    _sampled_pooled_correlation,
    summarize_predictions,
)


def test_exact_reconstruction_and_undefined_baselines():
    gene_ids = np.repeat(np.arange(2, dtype=np.int64), 6)
    counts = np.tile(np.array([0, 1, 2, 3, 4, 5], dtype=np.float64), 2)
    expected = np.log1p(counts)
    metrics, rows = summarize_predictions(
        gene_ids=gene_ids,
        counts=counts,
        logits=np.full(12, 30.0),
        intensity=expected,
        baseline_presence=np.full(12, 0.5),
        baseline_intensity=np.full(12, 0.7),
    )
    assert abs(metrics["model"]["positive_intensity_log1p"]["pearson"] - 1.0) < 1e-12
    assert abs(metrics["model"]["positive_intensity_log1p"]["spearman"] - 1.0) < 1e-12
    assert abs(metrics["model"]["positive_intensity_log1p"]["macro_gene_pearson"] - 1.0) < 1e-12
    assert abs(metrics["model"]["full_reconstruction_log1p"]["macro_gene_spearman"] - 1.0) < 1e-12
    assert metrics["model"]["full_reconstruction_log1p"]["mae"] < 1e-11
    assert metrics["baseline"]["positive_intensity_log1p"]["macro_gene_pearson"] is None
    assert metrics["baseline"]["full_reconstruction_log1p"]["macro_gene_spearman"] is None
    assert len(rows) == 2
    assert all(r["model_full_pearson"] is not None for r in rows)
    print("Known perfect correlations and undefined constant baselines: PASS")


def test_ties_sign_and_boundaries():
    x = np.array([0, 0, 1, 1, 2, 2], dtype=float)
    y = -x
    assert abs(_safe_pearson(x, y) + 1.0) < 1e-12
    assert abs(_safe_spearman(x, y) + 1.0) < 1e-12
    assert _safe_pearson(np.array([1, 2]), np.array([2, 3])) is None
    assert _safe_spearman(np.ones(5), np.arange(5)) is None
    only_zero, per_gene = summarize_predictions(
        gene_ids=np.zeros(4, dtype=np.int64),
        counts=np.zeros(4),
        logits=np.zeros(4),
        intensity=np.ones(4),
        baseline_presence=np.full(4, 0.2),
        baseline_intensity=np.zeros(4),
    )
    assert only_zero["model"]["positive_intensity_log1p"]["pearson"] is None
    assert only_zero["model"]["positive_intensity_log1p"]["macro_gene_spearman"] is None
    assert only_zero["model"]["full_reconstruction_log1p"]["pearson"] is None
    assert per_gene[0]["model_positive_pearson"] is None
    print("Ties, negative correlations and all-zero labels: PASS")


def test_deterministic_large_sample():
    values = np.linspace(0, 1, 400)
    transformed = values ** 2
    summary_a = _sampled_pooled_correlation(values, transformed, max_spearman_targets=75)
    summary_b = _sampled_pooled_correlation(values, transformed, max_spearman_targets=75)
    assert summary_a == summary_b
    assert summary_a["pearson_targets"] == 400
    assert summary_a["spearman_sampled_targets"] == 75
    assert summary_a["spearman_exact"] is False
    assert math.isclose(summary_a["spearman"], 1.0, abs_tol=1e-12)
    print("Sampled pooled Spearman reproducibility: PASS")


def main():
    test_exact_reconstruction_and_undefined_baselines()
    test_ties_sign_and_boundaries()
    test_deterministic_large_sample()
    print("All correlation metric tests passed.")


if __name__ == "__main__":
    main()
