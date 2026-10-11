"""Fast synthetic integration and resume tests for SG-MAE V1.0 training."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from src.data.gene_vocabulary import GeneVocabulary
from src.data.splitting import (
    load_spatial_split,
    save_spatial_split,
    split_spatial_validation,
)
from src.dataset import SGMAECellDataset
from src.models.gene_aware_mae import GeneAwareMAE
from src.trainer import (
    SGMAETrainer,
    TrainerConfig,
    load_best_weights,
    make_shaml_loaders,
    seed_training,
)
from src.training.losses import MaskedReconstructionLoss


FAKE_H5_SHA = "c" * 64


def fixture():
    rng = np.random.default_rng(17)
    genes = [f"ENSMUSG{i:011d}" for i in range(16)]
    vocabulary = GeneVocabulary.from_genes(
        genes, species="Mus musculus", namespace="ensembl_gene_id"
    )
    cell_ids = [f"cell_{i:03d}" for i in range(72)]
    expression = sparse.csr_matrix(rng.poisson(0.55, size=(72, 16)).astype(np.float32))
    dataset = SGMAECellDataset(
        expression, cell_ids, genes, [f"G{i}" for i in range(16)],
        vocabulary, "ShamL-synthetic",
    )
    frame = pd.DataFrame({
        "cell_id": cell_ids,
        "x_centroid": np.tile(np.arange(12) * 40, 6).astype(float),
        "y_centroid": np.repeat(np.arange(6) * 40, 12).astype(float),
    })
    split = split_spatial_validation(
        cell_ids, frame, sample_name="ShamL",
        block_size_um=40, buffer_um=0,
        validation_fraction=0.25, seed=42,
    )
    return dataset, vocabulary, frame, split


def setup_trainer(dataset, vocabulary, split, run_dir, *, epochs=3, resume=False, patience=10, min_delta=0.0):
    config = TrainerConfig(
        max_epochs=epochs, batch_size=9, learning_rate=0.001,
        weight_decay=0.01, mask_ratio=0.2,
        grad_clip_norm=1.0, patience=patience,
        min_delta=min_delta, seed=42, validation_mask_seed=77,
    )
    seed_training(config.seed)
    model_config = {
        "vocab_size": len(vocabulary),
        "token_dim": 32, "cell_dim": 32,
        "num_latents": 4, "num_heads": 4,
        "dropout": 0.1,
    }
    model = GeneAwareMAE(**model_config)
    loss = MaskedReconstructionLoss()
    training, validation, generator = make_shaml_loaders(dataset, split, config)
    return SGMAETrainer(
        model=model, criterion=loss, train_loader=training,
        validation_loader=validation, shuffle_generator=generator,
        config=config, model_config=model_config,
        vocabulary_fingerprint=vocabulary.fingerprint,
        split_fingerprint=split.fingerprint,
        data_sha256=FAKE_H5_SHA,
        run_dir=run_dir, device="cpu", resume=resume,
    )


def expect_raises(error, fn):
    try:
        fn()
    except error:
        return
    raise AssertionError(f"Expected {error.__name__}")


def main():
    torch.set_num_threads(1)
    dataset, vocabulary, frame, split = fixture()
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        frozen = root / "shaml_spatial_v1.json.gz"
        save_spatial_split(split, frozen)
        reloaded = load_spatial_split(frozen, dataset.cell_ids, frame)
        assert reloaded.fingerprint == split.fingerprint
        assert reloaded.counts == split.counts
        print("Frozen ShamL split verification: PASS")

        full = setup_trainer(dataset, vocabulary, reloaded, root / "full")
        first_val = full.evaluate_validation()
        second_val = full.evaluate_validation()
        assert first_val == second_val
        assert first_val["num_masked"] > 0
        print("Fixed validation masks and aggregated counts: PASS")

        # Reset global RNG because the preceding loader construction uses it.
        full = setup_trainer(dataset, vocabulary, reloaded, root / "continuous")
        full_history = full.fit()
        assert len(full_history) == 3
        assert full.best_path.is_file() and full.last_path.is_file()
        best = torch.load(full.best_path, map_location="cpu", weights_only=True)
        latest = torch.load(full.last_path, map_location="cpu", weights_only=True)
        assert best["kind"] == "best" and latest["kind"] == "last"
        assert latest["completed_epoch"] == 3
        assert latest["metadata"]["vocabulary_fingerprint"] == vocabulary.fingerprint
        assert latest["metadata"]["split_fingerprint"] == split.fingerprint
        print("Complete epoch training and provenance checkpoints: PASS")

        shadow = GeneAwareMAE(**full.metadata["model_config"])
        loaded = load_best_weights(shadow, full.best_path, full.metadata)
        assert loaded["best_epoch"] == full.best_epoch
        bad_metadata = {**full.metadata, "split_fingerprint": "0" * 64}
        expect_raises(ValueError, lambda: load_best_weights(shadow, full.best_path, bad_metadata))
        print("Best-checkpoint loading and provenance rejection: PASS")

        staged = setup_trainer(dataset, vocabulary, reloaded, root / "staged", epochs=1)
        staged.fit()
        resumed = setup_trainer(
            dataset, vocabulary, reloaded, root / "staged", epochs=3, resume=True
        )
        resumed_history = resumed.fit()
        assert len(resumed_history) == 3
        for key, value in full.model.state_dict().items():
            torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)
        for a, b in zip(full_history, resumed_history):
            assert a == b
        print("Exact CPU resume vs uninterrupted training: PASS")

        expect_raises(
            FileExistsError,
            lambda: setup_trainer(dataset, vocabulary, reloaded, root / "staged", epochs=4),
        )
        expect_raises(
            FileNotFoundError,
            lambda: setup_trainer(dataset, vocabulary, reloaded, root / "missing", resume=True),
        )
        print("Accidental checkpoint overwrite and missing resume rejected: PASS")

        wrong_config = replace(full.config, batch_size=8)
        seed_training(42)
        model = GeneAwareMAE(**full.metadata["model_config"])
        criterion = MaskedReconstructionLoss()
        a, b, shuffle = make_shaml_loaders(dataset, reloaded, wrong_config)
        expect_raises(ValueError, lambda: SGMAETrainer(
            model=model, criterion=criterion, train_loader=a, validation_loader=b,
            shuffle_generator=shuffle, config=wrong_config,
            model_config=full.metadata["model_config"],
            vocabulary_fingerprint=vocabulary.fingerprint,
            split_fingerprint=split.fingerprint, data_sha256=FAKE_H5_SHA,
            run_dir=root / "staged", device="cpu", resume=True,
        ))
        print("Resume with changed training configuration rejected: PASS")

        dataset.sample_name = "ShamR-synthetic"
        expect_raises(
            ValueError,
            lambda: make_shaml_loaders(dataset, reloaded, full.config),
        )
        dataset.sample_name = "ShamL-synthetic"
        print("ShamR and non-ShamL training rejected: PASS")

        stop_early = setup_trainer(
            dataset, vocabulary, reloaded, root / "early_stop",
            epochs=8, patience=2, min_delta=10000.0,
        )
        early_history = stop_early.fit()
        assert len(early_history) == 3
        assert stop_early.best_epoch == 1
        print("Early stopping on non-improvement: PASS")

    print("\nAll SG-MAE trainer integration tests passed.")


if __name__ == "__main__":
    main()
