"""Regression tests for read-only SG-MAE V1.0 evaluation.

Run from the project root: python -m tests.v1.test_evaluation
Synthetic data only. Never opens the project's real ShamR dataset.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch
from scipy import sparse

from src.data.gene_vocabulary import GeneVocabulary
from src.dataset import SGMAECellDataset
from src.evaluation.evaluate_sgmae_v1 import (
    evaluate_frozen_reconstruction,
    fit_train_only_baseline,
    load_frozen_best,
    summarize_predictions,
)
from src.models.gene_aware_mae import GeneAwareMAE
from src.trainer import SGMAETrainer, TrainerConfig, make_shaml_loaders
from src.training.losses import MaskedReconstructionLoss


def _fixture():
    vocabulary = GeneVocabulary.from_genes(
        [f"ENSMUSG{i:011d}" for i in range(7)],
        namespace="ensembl_gene_id",
    )
    values = np.array([
        [0, 2, 0, 1, 0, 0, 3],
        [0, 1, 0, 0, 0, 1, 2],
        [0, 0, 3, 0, 0, 1, 0],
        [1, 2, 0, 4, 0, 0, 0],
        [0, 4, 0, 0, 0, 0, 0],
        [1, 0, 2, 0, 0, 3, 0],
        [0, 2, 3, 0, 0, 0, 4],
        [1, 0, 0, 1, 0, 1, 0],
        [0, 3, 0, 0, 0, 0, 0],
        [0, 0, 2, 1, 0, 0, 4],
        [0, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 2, 0],
    ], dtype=np.float32)
    dataset = SGMAECellDataset(
        sparse.csr_matrix(values),
        [f"cell-{i:03d}" for i in range(values.shape[0])],
        list(vocabulary.genes),
        list(vocabulary.genes),
        vocabulary,
        "ShamL-synthetic",
    )
    train_indices = np.arange(0, 7)
    validation_indices = np.arange(7, 12)
    model = GeneAwareMAE(
        vocab_size=len(vocabulary),
        token_dim=16,
        cell_dim=16,
        num_latents=4,
        num_heads=4,
        dropout=0.0,
    )
    return vocabulary, dataset, train_indices, validation_indices, model


def test_train_only_baseline():
    _, dataset, train_indices, _, _ = _fixture()
    base = fit_train_only_baseline(dataset, train_indices)
    assert base["num_train_cells"] == 7
    # The 5th synthetic gene is never detected and uses Laplace smoothing.
    assert base["positive_counts"][4] == 0
    assert abs(base["prevalence"][4] - 1 / 9) < 1e-12
    assert base["positive_log_intensity"][4] == 0
    # Poisoning labels outside the training subset must not change the baseline.
    copy = dataset.expression.tolil(copy=True)
    copy[7:, :] = 9000
    dataset.expression = copy.tocsr()
    other = fit_train_only_baseline(dataset, train_indices)
    np.testing.assert_array_equal(base["prevalence"], other["prevalence"])
    np.testing.assert_array_equal(base["positive_log_intensity"], other["positive_log_intensity"])
    print("Train-only baseline and no validation leakage: PASS")


def test_metrics_and_zero_cases():
    metrics, genes = summarize_predictions(
        gene_ids=np.array([0, 0, 1, 1, 2, 2]),
        counts=np.array([0, 2, 0, 1, 0, 0]),
        logits=np.array([-4, 4, -2, 3, -3, -3], dtype=np.float32),
        intensity=np.array([0.1, 1.0, 0.1, 0.7, 0.1, 0.1]),
        baseline_presence=np.array([0.4, 0.4, 0.4, 0.4, 0.1, 0.1]),
        baseline_intensity=np.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.0]),
    )
    assert metrics["model"]["presence"]["auroc"] == 1.0
    assert metrics["model"]["presence"]["auprc"] == 1.0
    assert metrics["model"]["presence"]["f1_at_0_5"] == 1.0
    assert metrics["model"]["presence"]["auprc"] > metrics["baseline"]["presence"]["auprc"]
    assert len(genes) == 3 and genes[2]["model_auroc"] is None
    zeros, _ = summarize_predictions(
        gene_ids=np.array([0, 1]),
        counts=np.array([0, 0]),
        logits=np.array([-1.0, -2.0]),
        intensity=np.array([0.0, 0.0]),
        baseline_presence=np.array([0.2, 0.2]),
        baseline_intensity=np.array([0.0, 0.0]),
    )
    assert zeros["positive_targets"] == 0
    assert zeros["model"]["presence"]["auroc"] is None
    assert zeros["model"]["presence"]["auprc"] is None
    assert zeros["model"]["positive_intensity_log1p"]["mae"] is None
    print("AUROC/AUPRC/F1 and all-zero boundaries: PASS")


def test_undefined_ap_and_invalid_metric_inputs():
    # A gene with only positive targets cannot be assessed for ranking quality.
    summary, genes = summarize_predictions(
        gene_ids=np.array([0, 0, 1, 1]),
        counts=np.array([0, 3, 1, 2]),
        logits=np.array([-1.0, 1.0, -1.0, 1.0]),
        intensity=np.array([0.0, 1.0, 1.0, 1.0]),
        baseline_presence=np.array([0.5] * 4),
        baseline_intensity=np.array([0.5] * 4),
    )
    assert genes[0]["model_auprc"] is not None
    assert genes[1]["model_auprc"] is None
    assert genes[1]["model_auroc"] is None
    assert genes[1]["negative_targets"] == 0
    assert summary["model"]["presence"]["genes_with_auprc"] == 1

    all_positive, _ = summarize_predictions(
        gene_ids=np.array([0, 0]),
        counts=np.array([1, 2]),
        logits=np.array([-1.0, 1.0]),
        intensity=np.array([0.5, 0.5]),
        baseline_presence=np.array([0.5, 0.5]),
        baseline_intensity=np.array([0.5, 0.5]),
    )
    assert all_positive["model"]["presence"]["auprc"] is None

    arguments = {
        "gene_ids": np.array([0, 0]),
        "counts": np.array([0, 1]),
        "logits": np.array([-1.0, 1.0]),
        "intensity": np.array([0.1, 0.8]),
        "baseline_presence": np.array([0.5, 0.5]),
        "baseline_intensity": np.array([0.5, 0.5]),
    }
    for beta in (float("nan"), float("inf"), 0.0, -1.0):
        try:
            summarize_predictions(**arguments, huber_beta=beta)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid Huber beta accepted: {beta}")
    try:
        summarize_predictions(**{**arguments, "gene_ids": np.array([0.2, 0.8])})
    except TypeError:
        pass
    else:
        raise AssertionError("Fractional gene IDs were accepted.")
    print("Undefined AP and invalid evaluation inputs: PASS")


def test_frozen_mask_and_loss():
    torch.manual_seed(17)
    _, dataset, train_indices, val_indices, model = _fixture()
    baseline = fit_train_only_baseline(dataset, train_indices)
    cfg = {"batch_size": 2, "mask_ratio": 0.30, "validation_mask_seed": 314159}
    loss_cfg = {"lambda_presence": 1.0, "lambda_intensity": 1.0, "huber_beta": 1.0}
    stats_a, rows_a = evaluate_frozen_reconstruction(
        model=model,
        dataset=dataset,
        cell_indices=val_indices,
        baseline=baseline,
        config=cfg,
        loss_config=loss_cfg,
        show_progress=False,
    )
    stats_b, rows_b = evaluate_frozen_reconstruction(
        model=model,
        dataset=dataset,
        cell_indices=val_indices,
        baseline=baseline,
        config=cfg,
        loss_config=loss_cfg,
        show_progress=False,
    )
    assert stats_a == stats_b
    assert rows_a == rows_b
    assert stats_a["masked_targets"] == len(val_indices) * 2
    assert stats_a["num_cells"] == len(val_indices)
    reconstructed = (
        stats_a["model"]["presence"]["bce"]
        + stats_a["model"]["positive_intensity_log1p"]["huber"]
    )
    assert abs(stats_a["reconstruction_loss"]["total"] - reconstructed) < 1e-6
    print("Fixed validation masks and exact loss aggregation: PASS")


def test_trainer_validation_equivalence():
    torch.manual_seed(22)
    vocabulary, dataset, train_indices, val_indices, model = _fixture()
    split = SimpleNamespace(
        sample_name="ShamL",
        cell_ids=tuple(dataset.cell_ids),
        train_indices=train_indices,
        validation_indices=val_indices,
        excluded_indices=np.array([], dtype=np.int64),
        counts={"train": len(train_indices), "validation": len(val_indices), "excluded": 0},
    )
    config = TrainerConfig(
        max_epochs=1, batch_size=2,
        mask_ratio=0.3, validation_mask_seed=314159,
    )
    train_loader, validation_loader, shuffle_generator = make_shaml_loaders(dataset, split, config)
    criterion = MaskedReconstructionLoss()
    with TemporaryDirectory() as directory:
        trainer = SGMAETrainer(
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            validation_loader=validation_loader,
            shuffle_generator=shuffle_generator,
            config=config,
            model_config={"vocab_size": len(vocabulary), "token_dim": 16, "cell_dim": 16,
                          "num_latents": 4, "num_heads": 4, "dropout": 0.0},
            vocabulary_fingerprint=vocabulary.fingerprint,
            split_fingerprint="b" * 64,
            data_sha256="c" * 64,
            run_dir=directory,
            device="cpu",
        )
        actual = trainer.evaluate_validation()
        baseline = fit_train_only_baseline(dataset, train_indices)
        reproduced, _ = evaluate_frozen_reconstruction(
            model=model,
            dataset=dataset,
            cell_indices=val_indices,
            baseline=baseline,
            config={"batch_size": 2, "mask_ratio": 0.3, "validation_mask_seed": 314159},
            loss_config={"lambda_presence": 1.0, "lambda_intensity": 1.0, "huber_beta": 1.0},
            show_progress=False,
        )
        assert actual["num_masked"] == reproduced["masked_targets"]
        assert actual["num_positive"] == reproduced["positive_targets"]
        assert abs(actual["total_loss"] - reproduced["reconstruction_loss"]["total"]) < 1e-7
    print("Trainer validation and evaluator parity: PASS")


def test_checkpoint_provenance():
    vocab, _, _, _, model = _fixture()
    fingerprint = "a" * 64
    split_fingerprint = "b" * 64
    source_hash = "c" * 64
    payload = {
        "format_version": 1,
        "kind": "best",
        "best_epoch": 3,
        "best_validation": 0.251,
        "metadata": {
            "sample_name": "ShamL",
            "vocabulary_fingerprint": fingerprint,
            "split_fingerprint": split_fingerprint,
            "data_sha256": source_hash,
            "model_config": {
                "vocab_size": len(vocab), "token_dim": 16, "cell_dim": 16,
                "num_latents": 4, "num_heads": 4, "dropout": 0.0,
            },
            "trainer_config": {"batch_size": 2, "mask_ratio": 0.3, "validation_mask_seed": 314159},
            "loss_config": {"lambda_presence": 1.0, "lambda_intensity": 1.0, "huber_beta": 1.0},
        },
        "model_state_dict": model.state_dict(),
    }
    with TemporaryDirectory() as directory:
        path = Path(directory) / "best.pt"
        torch.save(payload, path)
        restored, loaded, metadata = load_frozen_best(
            path,
            vocabulary_fingerprint=fingerprint,
            split_fingerprint=split_fingerprint,
            sham_h5_sha256=source_hash,
            vocabulary_size=len(vocab),
        )
        assert loaded["best_epoch"] == 3
        for key, value in model.state_dict().items():
            assert torch.equal(value, restored.state_dict()[key])
        try:
            load_frozen_best(
                path,
                vocabulary_fingerprint="x" * 64,
                split_fingerprint=split_fingerprint,
                sham_h5_sha256=source_hash,
                vocabulary_size=len(vocab),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("Wrong vocabulary fingerprint was accepted.")
        payload["kind"] = "last"
        torch.save(payload, path)
        try:
            load_frozen_best(
                path,
                vocabulary_fingerprint=fingerprint,
                split_fingerprint=split_fingerprint,
                sham_h5_sha256=source_hash,
                vocabulary_size=len(vocab),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("last.pt was incorrectly accepted.")
    print("Best-checkpoint provenance and last.pt rejection: PASS")


def main():
    test_train_only_baseline()
    test_metrics_and_zero_cases()
    test_undefined_ap_and_invalid_metric_inputs()
    test_frozen_mask_and_loss()
    test_trainer_validation_equivalence()
    test_checkpoint_provenance()
    print("\nAll SG-MAE V1 evaluation tests passed.")


if __name__ == "__main__":
    main()
