"""Synthetic end-to-end CLI smoke tests; no real Xenium holdout is read.

Run from the project root: python -m tests.v1.test_evaluation_entrypoint
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from src.data.gene_vocabulary import GeneVocabulary
from src.data.splitting import save_spatial_split, split_spatial_validation
from src.dataset import load_sgmae_sample
from src.evaluation.evaluate_sgmae_v1 import evaluate_frozen_reconstruction, fit_train_only_baseline, sha256_file
from src.models.gene_aware_mae import GeneAwareMAE
from scripts.v1.evaluate import main as run_cli


def write_fake_xenium(path, counts, genes):
    """Write minimal valid Xenium CSC counts [genes, cells] for CLI testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    csc = sparse.csc_matrix(counts.T.astype(np.float32))
    with h5py.File(path, "w") as handle:
        matrix = handle.create_group("matrix")
        matrix.create_dataset("shape", data=np.asarray(csc.shape, dtype=np.int64))
        matrix.create_dataset("data", data=csc.data)
        matrix.create_dataset("indices", data=csc.indices)
        matrix.create_dataset("indptr", data=csc.indptr)
        matrix.create_dataset("barcodes", data=np.asarray([
            f"synthetic-cell-{i:03d}".encode() for i in range(counts.shape[0])
        ]))
        features = matrix.create_group("features")
        features.create_dataset("id", data=np.asarray([g.encode() for g in genes]))
        features.create_dataset("name", data=np.asarray([g.encode() for g in genes]))
        features.create_dataset("feature_type", data=np.asarray([b"Gene Expression"] * len(genes)))


def main():
    torch.manual_seed(23)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        genes = [f"ENSMUSG{i:011d}" for i in range(6)]
        vocabulary = GeneVocabulary.from_genes(genes, namespace="ensembl_gene_id")
        vocab_path = root / "configs" / "gene_vocabulary.json"
        vocab_path.parent.mkdir(parents=True)
        # The synthetic test uses the real fingerprinted vocabulary format.
        vocabulary.save(vocab_path)
        sham_l_path = root / "data" / "ShamL-output-fake" / "cell_feature_matrix.h5"
        sham_r_path = root / "data" / "ShamR-output-fake" / "cell_feature_matrix.h5"
        counts_l = np.random.default_rng(12).poisson(1.8, size=(24, 6)).astype(np.float32)
        counts_r = np.random.default_rng(13).poisson(1.2, size=(9, 6)).astype(np.float32)
        write_fake_xenium(sham_l_path, counts_l, genes)
        write_fake_xenium(sham_r_path, counts_r, genes)
        (sham_l_path.parent / "cells.parquet").touch()
        dataset = load_sgmae_sample(sham_l_path, vocabulary)
        # Three spatial groups without inter-block edges.
        frame = pd.DataFrame({
            "cell_id": dataset.cell_ids,
            "x_centroid": np.repeat([0.0, 400.0, 800.0], 8) + np.arange(24) % 8,
            "y_centroid": np.arange(24) % 8,
        })
        spatial = split_spatial_validation(
            dataset.cell_ids, frame, validation_fraction=0.25,
            block_size_um=250, seed=42, buffer_um=0.0, sample_name="ShamL",
        )
        manifest = root / "configs" / "splits" / "shaml_spatial_v1.json.gz"
        save_spatial_split(spatial, manifest)
        baseline = fit_train_only_baseline(dataset, spatial.train_indices)
        cfg = {"batch_size": 4, "mask_ratio": 0.5, "validation_mask_seed": 314159}
        loss_cfg = {"lambda_presence": 1.0, "lambda_intensity": 1.0, "huber_beta": 1.0}
        config = {"vocab_size": len(vocabulary), "token_dim": 16, "cell_dim": 16,
                  "num_latents": 4, "num_heads": 4, "dropout": 0.0}
        model = GeneAwareMAE(**config)
        expected, _ = evaluate_frozen_reconstruction(
            model=model,
            dataset=dataset,
            cell_indices=spatial.validation_indices,
            baseline=baseline,
            config=cfg,
            loss_config=loss_cfg,
            show_progress=False,
        )
        metadata = {
            "sample_name": "ShamL",
            "vocabulary_fingerprint": vocabulary.fingerprint,
            "split_fingerprint": spatial.fingerprint,
            "data_sha256": sha256_file(sham_l_path),
            "model_config": config,
            "trainer_config": cfg,
            "loss_config": loss_cfg,
        }
        ckpt_path = root / "runs" / "a" / "best.pt"
        ckpt_path.parent.mkdir(parents=True)
        torch.save({
            "format_version": 1,
            "kind": "best",
            "metadata": metadata,
            "best_epoch": 7,
            "best_validation": expected["reconstruction_loss"]["total"],
            "model_state_dict": model.state_dict(),
            "history": [{"validation": {
                "num_masked": expected["masked_targets"],
                "num_positive": expected["positive_targets"],
            }}],
        }, ckpt_path)
        out = root / "eval_shaml"
        common = [
            "--data-root", str(root / "data"),
            "--vocabulary", str(vocab_path),
            "--split", str(manifest),
            "--device", "cpu",
            "--checkpoint", f"fixed={ckpt_path}",
            "--checkpoint", f"schedule={ckpt_path}",
        ]
        def read_fake_parquet(path, columns):
            assert Path(path).parent == sham_l_path.parent
            return frame[columns].copy()
        with patch("pandas.read_parquet", side_effect=read_fake_parquet):
            report = run_cli(common + ["--output-dir", str(out)])
            assert out.joinpath("summary.json").is_file()
            assert out.joinpath("fixed_per_gene.csv").is_file()
            assert out.joinpath("schedule_per_gene.csv").is_file()
            assert report["results"]["fixed"]["mask_sha256"] == report["results"]["schedule"]["mask_sha256"]
            try:
                run_cli(common + ["--output-dir", str(out)])
            except FileExistsError:
                pass
            else:
                raise AssertionError("Existing evaluation directory was overwritten.")
        print("ShamL CLI, two checkpoints and read-only reports: PASS")
        with patch("pandas.read_parquet", side_effect=read_fake_parquet):
            try:
                run_cli(common + ["--sample", "shamr", "--output-dir", str(root / "eval_shamr")])
            except SystemExit as error:
                assert error.code != 0
            else:
                raise AssertionError("Held-out ShamR was opened without confirmation.")
            shamr = run_cli(common + [
                "--sample", "shamr", "--confirm-heldout",
                "--output-dir", str(root / "eval_shamr"),
            ])
            assert shamr["heldout"] and shamr["results"]["fixed"]["num_cells"] == 9
        print("Explicit held-out access and ShamR synthetic evaluation: PASS")
    print("\nAll SG-MAE V1 evaluation entrypoint tests passed.")


if __name__ == "__main__":
    main()
