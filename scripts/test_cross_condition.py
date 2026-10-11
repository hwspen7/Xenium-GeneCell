"""Synthetic four-slice regression tests for the cross-condition evaluator.

Does not open or alter the user's real Xenium datasets or checkpoints.
"""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from scripts.evaluate_sgmae_conditions import main as run_four_slices
from scripts.evaluate_sgmae_v1 import main as run_one_slice
from scripts.test_evaluation_entrypoint import write_fake_xenium
from src.data.gene_vocabulary import GeneVocabulary
from src.data.splitting import save_spatial_split, split_spatial_validation
from src.dataset import load_sgmae_sample
from src.evaluation.evaluate_sgmae_v1 import (
    evaluate_frozen_reconstruction,
    fit_train_only_baseline,
    sha256_file,
)
from src.models.gene_aware_mae import GeneAwareMAE


def main():
    torch.manual_seed(23)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        genes = [f"ENSMUSG{i:011d}" for i in range(6)]
        vocabulary = GeneVocabulary.from_genes(genes, namespace="ensembl_gene_id")
        vocab_path = root / "configs" / "gene_vocabulary.json"
        vocabulary.save(vocab_path)

        counts = {
            "ShamL": np.random.default_rng(12).poisson(1.8, size=(24, 6)).astype(np.float32),
            "ShamR": np.random.default_rng(13).poisson(1.2, size=(9, 6)).astype(np.float32),
            "Hour12R": np.random.default_rng(14).poisson(1.5, size=(11, 6)).astype(np.float32),
            "Day2L": np.random.default_rng(15).poisson(0.9, size=(13, 6)).astype(np.float32),
        }
        for prefix, expression in counts.items():
            ids = genes
            if prefix == "Day2L":
                permutation = [5, 4, 3, 2, 1, 0]
                ids = [genes[i] for i in permutation]
                expression = expression[:, permutation]
            path = root / "data" / f"{prefix}-output-fake" / "cell_feature_matrix.h5"
            write_fake_xenium(path, expression, ids)

        sham_l_path = root / "data" / "ShamL-output-fake" / "cell_feature_matrix.h5"
        (sham_l_path.parent / "cells.parquet").touch()
        dataset = load_sgmae_sample(sham_l_path, vocabulary)
        frame = pd.DataFrame({
            "cell_id": dataset.cell_ids,
            "x_centroid": np.repeat([0.0, 400.0, 800.0], 8) + np.arange(24) % 8,
            "y_centroid": np.arange(24) % 8,
        })
        spatial = split_spatial_validation(
            dataset.cell_ids, frame, validation_fraction=0.25,
            block_size_um=250, seed=42, buffer_um=0.0, sample_name="ShamL",
        )
        split_path = root / "configs" / "splits" / "shaml_spatial_v1.json.gz"
        save_spatial_split(spatial, split_path)
        baseline = fit_train_only_baseline(dataset, spatial.train_indices)
        cfg = {"batch_size": 4, "mask_ratio": 0.5, "validation_mask_seed": 314159}
        loss_cfg = {"lambda_presence": 1.0, "lambda_intensity": 1.0, "huber_beta": 1.0}
        config = {"vocab_size": len(vocabulary), "token_dim": 16, "cell_dim": 16,
                  "num_latents": 4, "num_heads": 4, "dropout": 0.0}
        model = GeneAwareMAE(**config)
        expected, _ = evaluate_frozen_reconstruction(
            model=model, dataset=dataset, cell_indices=spatial.validation_indices,
            baseline=baseline, config=cfg, loss_config=loss_cfg, show_progress=False,
        )
        ckpt_path = root / "runs" / "a" / "best.pt"
        ckpt_path.parent.mkdir(parents=True)
        torch.save({
            "format_version": 1,
            "kind": "best",
            "metadata": {
                "sample_name": "ShamL",
                "vocabulary_fingerprint": vocabulary.fingerprint,
                "split_fingerprint": spatial.fingerprint,
                "data_sha256": sha256_file(sham_l_path),
                "model_config": config,
                "trainer_config": cfg,
                "loss_config": loss_cfg,
            },
            "best_epoch": 7,
            "best_validation": expected["reconstruction_loss"]["total"],
            "model_state_dict": model.state_dict(),
            "history": [{"validation": {
                "num_masked": expected["masked_targets"],
                "num_positive": expected["positive_targets"],
            }}],
        }, ckpt_path)

        output_dir = root / "cross_condition"
        common = [
            "--data-root", str(root / "data"),
            "--vocabulary", str(vocab_path),
            "--split", str(split_path),
            "--device", "cpu",
            "--checkpoint", f"baseline={ckpt_path}",
            "--checkpoint", f"scheduler={ckpt_path}",
            "--output-dir", str(output_dir),
        ]
        try:
            run_four_slices(common)
        except SystemExit as error:
            assert error.code != 0
        else:
            raise AssertionError("External slices accessed without opt-in.")
        assert not output_dir.exists()
        print("External-slice confirmation: PASS")

        def read_fake_parquet(path, columns):
            assert Path(path).parent == sham_l_path.parent
            return frame[columns].copy()

        with patch("pandas.read_parquet", side_effect=read_fake_parquet):
            rows = run_four_slices(common + ["--confirm-external"])
        assert len(rows) == 12
        assert (output_dir / "comparison.csv").is_file()
        assert (output_dir / "four_slice_comparison.png").stat().st_size > 20000
        for prefix in ("shaml", "shamr", "hour12r", "day2l"):
            report = json.loads((output_dir / prefix / "summary.json").read_text())
            assert report["sample"] == prefix
            assert report["heldout"] == (prefix != "shaml")
            assert report["results"]["baseline"]["mask_sha256"] == report["results"]["scheduler"]["mask_sha256"]
        assert rows[0]["scope"] == "Validation"
        assert rows[-1]["scope"] == "External slice"
        assert rows[0]["method"] == "Gene-frequency baseline"
        assert rows[-1]["method"] == "scheduler"
        print("Four-slice evaluation and combined figure: PASS")

        with (output_dir / "comparison.csv").open(encoding="utf-8", newline="") as stream:
            saved_rows = list(csv.DictReader(stream))
        assert len(saved_rows) == 12
        for slice_id in ("shaml", "shamr", "hour12r", "day2l"):
            matching = [row for row in saved_rows if row["sample"] == slice_id]
            assert len(matching) == 3
            assert {row["method"] for row in matching} == {
                "Gene-frequency baseline", "baseline", "scheduler"
            }
        print("Matched frozen checkpoints and per-slice masks: PASS")

        try:
            run_four_slices(common + ["--confirm-external"])
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing report folder was overwritten.")
        print("No overwrite of prior reports: PASS")

        # The single-sample entrypoint must also evaluate injury explicitly.
        with patch("pandas.read_parquet", side_effect=read_fake_parquet):
            injury = run_one_slice([
                "--data-root", str(root / "data"),
                "--vocabulary", str(vocab_path),
                "--split", str(split_path),
                "--device", "cpu",
                "--checkpoint", f"baseline={ckpt_path}",
                "--sample", "hour12r", "--confirm-heldout",
                "--output-dir", str(root / "injury_single"),
            ])
        assert injury["timepoint"] == "12h"
        assert injury["evaluation_scope"] == "external_slice"
        print("Individual 12-hour evaluation: PASS")

    print("\nAll SG-MAE V1 cross-condition tests passed.")


if __name__ == "__main__":
    main()
