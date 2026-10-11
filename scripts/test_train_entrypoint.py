"""Entry-point smoke test with synthetic ShamL H5 and mocked Parquet IO.

This tests the real CLI main function without requiring pyarrow to be
installed in the test runner. Local production runs use actual Parquet IO.
"""

import contextlib
import io
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from src.data.gene_vocabulary import GeneVocabulary
from src.data.splitting import save_spatial_split, split_spatial_validation
from scripts.train_sgmae_v1 import main as train_main


def invoke(arguments, frame):
    output = io.StringIO()
    # Mimic the CLI's argument parsing while using an in-memory Parquet frame.
    with patch.object(sys, "argv", ["train_sgmae_v1.py", *arguments]):
        with patch("pandas.read_parquet", return_value=frame.copy()):
            with contextlib.redirect_stdout(output):
                train_main()
    return output.getvalue()


def main():
    import torch
    torch.set_num_threads(1)
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        sample_dir = root / "data" / "ShamL-synthetic"
        sample_dir.mkdir(parents=True)
        values = np.random.default_rng(123).poisson(0.7, size=(40, 16)).astype(np.float32)
        matrix = sparse.csc_matrix(values.T)
        genes = [f"ENSMUSG{i:011d}" for i in range(16)]
        ids = [f"cell_{i:03d}" for i in range(40)]
        with h5py.File(sample_dir / "cell_feature_matrix.h5", "w") as f:
            grp = f.create_group("matrix")
            grp.create_dataset("shape", data=np.array(matrix.shape))
            grp.create_dataset("data", data=matrix.data)
            grp.create_dataset("indices", data=matrix.indices)
            grp.create_dataset("indptr", data=matrix.indptr)
            grp.create_dataset("barcodes", data=np.array(ids, dtype="S"))
            feat = grp.create_group("features")
            feat.create_dataset("id", data=np.array(genes, dtype="S"))
            feat.create_dataset("name", data=np.array([f"G{i}" for i in range(16)], dtype="S"))
            feat.create_dataset("feature_type", data=np.array(["Gene Expression"] * 16, dtype="S"))
        frame = pd.DataFrame({
            "cell_id": [cid.encode() for cid in ids],
            "x_centroid": np.tile(np.arange(8) * 40, 5).astype(float),
            "y_centroid": np.repeat(np.arange(5) * 40, 8).astype(float),
        })
        # A stub exists only to exercise the normal Parquet-path control flow.
        (sample_dir / "cells.parquet").touch()
        vocab = GeneVocabulary.from_genes(
            genes, species="Mus musculus", namespace="ensembl_gene_id"
        )
        vocabulary_path = root / "configs" / "gene_vocabulary.json"
        vocab.save(vocabulary_path)
        canonical_ids = [cid.decode() for cid in frame["cell_id"]]
        split = split_spatial_validation(
            canonical_ids, frame, sample_name="ShamL",
            validation_fraction=0.25, block_size_um=40, buffer_um=0, seed=42,
        )
        split_path = root / "configs" / "splits" / "shaml_spatial_v1.json.gz"
        save_spatial_split(split, split_path)
        config_path = root / "configs" / "v1_training.json"
        payload = {
            "model": {"token_dim": 32, "cell_dim": 32, "num_latents": 4, "num_heads": 4, "dropout": 0.0},
            "loss": {"lambda_presence": 1.0, "lambda_intensity": 1.0, "huber_beta": 1.0},
            "trainer": {
                "max_epochs": 1, "batch_size": 8, "learning_rate": 0.001,
                "weight_decay": 0.01, "mask_ratio": 0.2, "grad_clip_norm": 1.0,
                "patience": 4, "min_delta": 0.0, "seed": 42, "validation_mask_seed": 77,
            },
            "expected_vocabulary_fingerprint": vocab.fingerprint,
            "expected_split_fingerprint": split.fingerprint,
        }
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        run_dir = root / "runs" / "v1"
        args = [
            "--config", str(config_path), "--data-root", str(root / "data"),
            "--vocabulary", str(vocabulary_path), "--split", str(split_path),
            "--run-dir", str(run_dir), "--device", "cpu",
        ]
        result = invoke(args + ["--dry-run"], frame)
        assert "DRY RUN" in result and not run_dir.exists()
        print("Entrypoint provenance-only dry run: PASS")

        result = invoke(args, frame)
        assert "Finished." in result
        assert (run_dir / "best.pt").is_file() and (run_dir / "last.pt").is_file()
        print("Entrypoint synthetic ShamL end-to-end training: PASS")

        result = invoke(args + ["--resume", "--epochs", "2"], frame)
        assert "Resumed after epoch 1" in result and "Epoch 002" in result
        print("Entrypoint checkpoint resume and extended epochs: PASS")

        payload["expected_split_fingerprint"] = "0" * 64
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            invoke(args + ["--dry-run"], frame)
        except ValueError as error:
            assert "Frozen spatial split differs" in str(error)
        else:
            raise AssertionError("Incorrect frozen split was accepted.")
        print("Entrypoint incorrect frozen split rejected: PASS")

    print("\nAll SG-MAE V1 entrypoint smoke tests passed.")


if __name__ == "__main__":
    main()
