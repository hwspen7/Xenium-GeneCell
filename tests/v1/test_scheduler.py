"""Test plateau LR reduction, exact resume, and fixed-LR compatibility."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from tests.v1.test_trainer import FAKE_H5_SHA, fixture
from src.models.gene_aware_mae import GeneAwareMAE
from src.trainer import (
    SGMAETrainer,
    SchedulerConfig,
    TrainerConfig,
    make_shaml_loaders,
    seed_training,
)
from src.training.losses import MaskedReconstructionLoss


def make_trainer(dataset, vocabulary, split, run_dir, *, epochs, resume=False, scheduler=True):
    # A deliberately large threshold makes LR reduction deterministic.
    options = {
        "name": "reduce_on_plateau",
        "factor": 0.5,
        "patience": 1,
        "threshold": 100.0,
        "threshold_mode": "abs",
        "cooldown": 0,
        "min_lr": 0.0001,
    } if scheduler else None
    config = TrainerConfig(
        max_epochs=epochs,
        batch_size=9,
        learning_rate=0.001,
        weight_decay=0.01,
        mask_ratio=0.2,
        grad_clip_norm=1.0,
        patience=20,
        min_delta=0.0,
        seed=42,
        validation_mask_seed=77,
        scheduler=options,
    )
    seed_training(config.seed)
    model_config = {
        "vocab_size": len(vocabulary),
        "token_dim": 32,
        "cell_dim": 32,
        "num_latents": 4,
        "num_heads": 4,
        "dropout": 0.1,
    }
    model = GeneAwareMAE(**model_config)
    train_loader, validation_loader, shuffle_generator = make_shaml_loaders(
        dataset, split, config
    )
    return SGMAETrainer(
        model=model,
        criterion=MaskedReconstructionLoss(),
        train_loader=train_loader,
        validation_loader=validation_loader,
        shuffle_generator=shuffle_generator,
        config=config,
        model_config=model_config,
        vocabulary_fingerprint=vocabulary.fingerprint,
        split_fingerprint=split.fingerprint,
        data_sha256=FAKE_H5_SHA,
        run_dir=run_dir,
        device="cpu",
        resume=resume,
    )


def expect_error(error_type, action):
    try:
        action()
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__}.")


def main():
    torch.set_num_threads(1)
    dataset, vocabulary, _, split = fixture()

    for invalid in (
        {"factor": 1.0},
        {"patience": -1},
        {"threshold_mode": "unknown"},
        {"min_lr": -0.1},
    ):
        expect_error(ValueError, lambda x=invalid: SchedulerConfig(**x))
    expect_error(ValueError, lambda: TrainerConfig(
        learning_rate=0.001, scheduler={"min_lr": 0.01}
    ))
    print("Scheduler configuration validation: PASS")

    with TemporaryDirectory() as directory:
        root = Path(directory)

        full = make_trainer(dataset, vocabulary, split, root / "full", epochs=5)
        history = full.fit()
        assert len(history) == 5
        assert [round(h["learning_rate"], 7) for h in history] == [
            0.001, 0.001, 0.001, 0.0005, 0.0005
        ]
        assert abs(history[-1]["next_learning_rate"] - 0.00025) < 1e-12
        assert abs(full.optimizer.param_groups[0]["lr"] - 0.00025) < 1e-12
        ckpt = torch.load(full.last_path, map_location="cpu", weights_only=True)
        assert "scheduler_state_dict" in ckpt
        assert "scheduler" in ckpt["metadata"]["trainer_config"]
        assert len(ckpt["history"]) == 5
        print("ReduceLROnPlateau reductions and LR history: PASS")

        stage = make_trainer(dataset, vocabulary, split, root / "stage", epochs=2)
        stage.fit()
        resumed = make_trainer(
            dataset, vocabulary, split, root / "stage", epochs=5, resume=True
        )
        resumed_history = resumed.fit()
        assert resumed_history == history
        for name, value in full.model.state_dict().items():
            torch.testing.assert_close(
                value, resumed.model.state_dict()[name], rtol=0, atol=0
            )
        assert full.scheduler.state_dict() == resumed.scheduler.state_dict()
        print("Exact CPU scheduler checkpoint resume: PASS")

        expect_error(ValueError, lambda: make_trainer(
            dataset, vocabulary, split, root / "stage", epochs=6,
            resume=True, scheduler=False,
        ))
        expect_error(ValueError, lambda: make_trainer(
            dataset, vocabulary, split, root / "full", epochs=6,
            resume=True, scheduler=False,
        ))
        print("Scheduler/fixed-LR checkpoint mixing rejected: PASS")

        missing = root / "missing_state"
        missing.mkdir()
        tampered = torch.load(full.last_path, map_location="cpu", weights_only=True)
        tampered.pop("scheduler_state_dict")
        torch.save(tampered, missing / "last.pt")
        expect_error(ValueError, lambda: make_trainer(
            dataset, vocabulary, split, missing, epochs=6, resume=True
        ))
        print("Missing scheduler resume state rejected: PASS")

        fixed = make_trainer(
            dataset, vocabulary, split, root / "fixed", epochs=2, scheduler=False
        )
        fixed.fit()
        fixed_ckpt = torch.load(fixed.last_path, map_location="cpu", weights_only=True)
        assert "scheduler_state_dict" not in fixed_ckpt
        assert "scheduler" not in fixed_ckpt["metadata"]["trainer_config"]
        fixed_resumed = make_trainer(
            dataset, vocabulary, split, root / "fixed", epochs=3,
            resume=True, scheduler=False,
        )
        fixed_resumed.fit()
        assert len(fixed_resumed.history) == 3
        print("Legacy fixed-LR workflow compatibility: PASS")

    print("\nAll SG-MAE scheduler tests passed.")


if __name__ == "__main__":
    main()
