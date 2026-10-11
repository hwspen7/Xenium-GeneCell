"""Synthetic regression tests for frozen cell representation QC and expression baselines."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from scripts.evaluate_cell_representations import _choose_rows, main
from src.evaluation.representation_diagnostics import (
    aligned_gene_columns,
    embedding_diagnostics,
    embedding_reference_pc1,
    expression_transform,
    finite_pearson,
    fit_train_reference_pca,
    matched_cell_rows,
    project_common_2d,
)


class FakeDataset:
    def __init__(self, counts, ids, sample):
        self.expression = sparse.csr_matrix(counts, dtype=np.float32)
        self.gene_ids = torch.as_tensor(ids, dtype=torch.long)
        self.cell_ids = [f"{sample}-{i}" for i in range(len(counts))]

    def __len__(self):
        return self.expression.shape[0]


def check_metrics():
    x = np.asarray([1., 2., 3., 4.])
    assert abs(finite_pearson(x, x) - 1.0) < 1e-12
    assert abs(finite_pearson(x, -x) + 1.0) < 1e-12
    assert finite_pearson(x, np.ones_like(x)) is None
    counts = np.array([[0, 0, 0], [2, 0, 3], [0, 5, 0]], dtype=np.float32)
    transformed = expression_transform(counts)
    assert np.array_equal(transformed[0], np.zeros(3))
    assert transformed[1, 1] == 0 and transformed[2, 0] == 0
    assert np.allclose(np.expm1(transformed[1]).sum(), 10000, rtol=0.0001)
    assert np.allclose(expression_transform(counts, method="raw_log1p"), np.log1p(counts))
    for bad in (np.array([[-1., 1.]]), np.array([[np.nan, 0.]])):
        try:
            expression_transform(bad)
            raise AssertionError("Invalid expression counts accepted")
        except ValueError:
            pass


def check_alignment():
    dataset = FakeDataset([[1, 0, 3], [4, 2, 0], [2, 6, 8]], [7, 3, 5], "shaml")
    cols = aligned_gene_columns(dataset, [3, 5, 7])
    assert cols.tolist() == [1, 2, 0]
    meta = pd.DataFrame({"sample": ["shaml", "shaml"],
                         "row_index": [2, 0], "cell_id": ["shaml-2", "shaml-0"]})
    assert matched_cell_rows(dataset, meta).tolist() == [2, 0]
    bad = meta.copy()
    bad.loc[0, "cell_id"] = "shaml-1"
    try:
        matched_cell_rows(dataset, bad)
        raise AssertionError("Mismatched H5 cells accepted")
    except ValueError:
        pass
    try:
        aligned_gene_columns(dataset, [3, 5, 8])
        raise AssertionError("Mismatched gene panel accepted")
    except ValueError:
        pass


def check_embedding_qc():
    rng = np.random.default_rng(43)
    latent = rng.normal(size=(100, 6)).astype(np.float32)
    counts = np.exp(latent[:, 0] * 0.5 + 3.0)
    detected = np.arange(100, dtype=np.int32)
    pc1 = embedding_reference_pc1(latent)
    result = embedding_diagnostics(latent, counts, detected, pc1_reference=pc1)
    assert result["cells"] == 100 and result["dimensions"] == 6
    assert 1.0 <= result["effective_rank"] <= 6.0001
    assert 0 <= result["top1_variance_fraction"] <= 1
    assert result["near_constant_dimensions_abs_std_lt_1e-6"] == 0
    assert -1 <= result["reference_pc1_r_log1p_counts"] <= 1
    collapse = np.ones((8, 4), dtype=np.float32)
    report = embedding_diagnostics(collapse, np.ones(8), np.ones(8),
                                   pc1_reference=(np.ones(4), np.ones(4) / 2))
    assert report["effective_rank"] == 0
    assert report["embedding_norm_r_log1p_counts"] is None
    model = fit_train_reference_pca(rng.poisson(3, size=(75, 8)), feature_dim=6, seed=42)
    assert model.n_components_ == 6
    pc, um, fractions = project_common_2d(latent, skip_umap=True)
    assert pc.shape == (100, 2) and um is None and sum(fractions) <= 1 + 1e-5


def check_full_synthetic_pipeline():
    rng = np.random.default_rng(88)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        input_dir = root / "embedded"
        input_dir.mkdir()
        names = ("shaml", "shamr", "hour12r", "day2l")
        prefix_names = ("ShamL", "ShamR", "Hour12R", "Day2L")
        datasets = {}
        data = {}
        reference_ids = np.array([2, 4, 6, 8, 10, 12, 14, 16])
        for i, (sample, prefix) in enumerate(zip(names, prefix_names)):
            n = 32 + i
            counts = rng.poisson(2.0 + i * 0.4, size=(n, 8)).astype(np.float32)
            if sample == "day2l":
                counts = counts[:, [2, 7, 0, 6, 3, 5, 4, 1]]
                gene_ids = reference_ids[[2, 7, 0, 6, 3, 5, 4, 1]]
            else:
                gene_ids = reference_ids
            datasets[prefix] = FakeDataset(counts, gene_ids, sample)
            embeddings = rng.normal(size=(n, 6)).astype(np.float32)
            meta = pd.DataFrame({
                "sample": [sample] * n,
                "row_index": np.arange(n, dtype=int),
                "cell_id": datasets[prefix].cell_ids,
                "total_counts": counts.sum(axis=1),
                "detected_genes": (counts > 0).sum(axis=1),
            })
            data[sample] = {"embedding": embeddings, "metadata": meta}
            folder = input_dir / sample
            folder.mkdir()
            (folder / "manifest.json").write_text(json.dumps({"source_h5_sha256": prefix}))
        provenance = {
            "vocabulary_fingerprint": "vocab",
            "split_fingerprint": "split",
            "shaml_h5_sha256": "ShamL",
            "checkpoint_sha256": "model",
            "encoding_protocol": "full_measured_visible_no_mask",
        }
        out_dir = root / "result"
        fake_vocab = SimpleNamespace(species="Mus musculus", namespace="ensembl_gene_id", fingerprint="vocab")
        fake_split = SimpleNamespace(train_indices=np.arange(24), fingerprint="split")
        def locator(_, prefix):
            return Path(prefix) / "cell_feature_matrix.h5"
        def loader(path, _):
            return datasets[path.parent.name]
        def digest(path):
            return path.parent.name
        arguments = ["--input-dir", str(input_dir), "--data-root", str(root),
                     "--output-dir", str(out_dir), "--per-sample", "8", "--skip-umap"]
        with patch("scripts.evaluate_cell_representations._read_data", return_value=(provenance, data)), \
             patch("scripts.evaluate_cell_representations.GeneVocabulary.load", return_value=fake_vocab), \
             patch("scripts.evaluate_cell_representations.find_xenium_sample_file", side_effect=locator), \
             patch("scripts.evaluate_cell_representations.load_sgmae_sample", side_effect=loader), \
             patch("scripts.evaluate_cell_representations.sha256_file", side_effect=digest), \
             patch("scripts.evaluate_cell_representations.load_spatial_split", return_value=fake_split):
            result = main(arguments)
        assert result["total_selected"] == 32
        assert result["expression_pca_components"] == 6
        assert (out_dir / "sample_qc.csv").is_file()
        assert (out_dir / "sgmae_vs_expression_pca.png").is_file()
        assert (out_dir / "embedding_numerical_qc.png").is_file()
        assert (out_dir / "matched_technical_associations.png").is_file()
        assert len(pd.read_csv(out_dir / "matched_representation_qc.csv")) == 8
        assert len(pd.read_csv(out_dir / "matched_projection_cells.csv")) == 32
        assert pd.read_csv(out_dir / "sample_qc.csv")["cells"].tolist() == [32, 33, 34, 35]
        assert result["expression_pca_fit"] == "ShamL training cells only"
        selected = pd.read_csv(out_dir / "matched_projection_cells.csv")
        for key in names:
            assert selected.loc[selected["sample"] == key, "row_index"].is_unique
        vectors, subset, per_slice = _choose_rows(data, projection_csv=out_dir / "matched_projection_cells.csv", per_sample=4, seed=12)
        assert len(vectors) == 32 and per_slice == 8
        bad = subset.copy()
        bad.loc[0, "cell_id"] = "not-a-real-cell"
        bad_path = root / "bad.csv"
        bad.to_csv(bad_path, index=False)
        try:
            _choose_rows(data, projection_csv=bad_path, per_sample=8, seed=42)
            raise AssertionError("Unknown selected Cell ID accepted")
        except ValueError:
            pass


def main_test():
    check_metrics()
    check_alignment()
    check_embedding_qc()
    check_full_synthetic_pipeline()
    print("Cell representation diagnostics and matched baseline: all tests passed")


if __name__ == "__main__":
    main_test()
