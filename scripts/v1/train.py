"""Train SG-MAE V1.0 on frozen ShamL training cells only.

Do not read ShamR, Hour12R, or Day2L in this entrypoint.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.data.gene_vocabulary import GeneVocabulary
from src.data.splitting import load_spatial_split
from src.dataset import find_xenium_sample_file, load_sgmae_sample
from src.models.gene_aware_mae import GeneAwareMAE
from src.trainer import (
    SGMAETrainer,
    TrainerConfig,
    make_shaml_loaders,
    seed_training,
)
from src.training.losses import MaskedReconstructionLoss


def sha256_file(path: Path) -> str:
    """Hash all input H5 bytes to detect changes before checkpoint resume."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Frozen ShamL-only SG-MAE trainer")
    parser.add_argument("--config", type=Path, default=root / "configs" / "v1" / "training.json")
    parser.add_argument("--data-root", type=Path, default=root / "data")
    parser.add_argument("--vocabulary", type=Path, default=root / "configs" / "gene_vocabulary.json")
    parser.add_argument("--split", type=Path, default=root / "configs" / "splits" / "shaml_spatial_v1.json.gz")
    parser.add_argument("--run-dir", type=Path, default=root / "runs" / "v1" / "training" / "new_experiment")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    settings = json.loads(args.config.read_text(encoding="utf-8"))
    expected_fields = {
        "model", "loss", "trainer", "expected_vocabulary_fingerprint",
        "expected_split_fingerprint",
    }
    if not isinstance(settings, dict) or set(settings) != expected_fields:
        raise ValueError("Invalid or incomplete V1 training configuration.")
    cfg_data = settings["trainer"].copy()
    if args.epochs is not None:
        cfg_data["max_epochs"] = args.epochs
    config = TrainerConfig(**cfg_data)
    seed_training(config.seed)

    vocabulary = GeneVocabulary.load(args.vocabulary)
    if vocabulary.namespace != "ensembl_gene_id" or vocabulary.species != "Mus musculus":
        raise ValueError("Expected mouse Ensembl Gene ID vocabulary.")
    if vocabulary.fingerprint != settings["expected_vocabulary_fingerprint"]:
        raise ValueError("Vocabulary differs from frozen V1 configuration.")

    sham_path = find_xenium_sample_file(args.data_root, "ShamL")
    dataset = load_sgmae_sample(sham_path, vocabulary)
    if not (dataset.sample_name == "ShamL" or dataset.sample_name.startswith("ShamL-")):
        raise ValueError("Refusing to train with any sample other than ShamL.")
    split = load_spatial_split(
        args.split,
        dataset.cell_ids,
        sham_path.parent / "cells.parquet",
        expected_sample_name="ShamL",
    )
    if split.fingerprint != settings["expected_split_fingerprint"]:
        raise ValueError("Frozen spatial split differs from V1 configuration.")
    if dataset.vocabulary_fingerprint != vocabulary.fingerprint:
        raise ValueError("Dataset/Vocabulary fingerprint mismatch.")
    data_sha256 = sha256_file(sham_path)

    print("SG-MAE V1.0 | ShamL-only training")
    print(f"Sample: {sham_path.parent.name}")
    print(f"Cells: train={split.counts['train']:,} validation={split.counts['validation']:,} excluded={split.counts['excluded']:,}")
    print(f"Vocabulary: {len(vocabulary):,} IDs | {vocabulary.fingerprint}")
    print(f"Frozen split: {split.fingerprint}")
    print(f"Input H5 SHA-256: {data_sha256}")
    print("ShamR, Hour12R, Day2L: not opened or used")
    if args.dry_run:
        print("DRY RUN: provenance and split verified. No training or checkpoint writes.")
        return

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    model_config = {"vocab_size": len(vocabulary), **settings["model"]}
    model = GeneAwareMAE(**model_config)
    criterion = MaskedReconstructionLoss(**settings["loss"])
    train_loader, val_loader, generator = make_shaml_loaders(dataset, split, config)
    trainer = SGMAETrainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        validation_loader=val_loader,
        shuffle_generator=generator,
        config=config,
        model_config=model_config,
        vocabulary_fingerprint=vocabulary.fingerprint,
        split_fingerprint=split.fingerprint,
        data_sha256=data_sha256,
        run_dir=args.run_dir,
        device=device,
        resume=args.resume,
    )
    print(f"Device: {device} | epochs: {config.max_epochs} | batch: {config.batch_size}")
    trainer.fit()
    print(f"Finished. Best validation epoch: {trainer.best_epoch}")
    print(f"Best checkpoint: {trainer.best_path}")
    print(f"Latest resume checkpoint: {trainer.last_path}")


if __name__ == "__main__":
    main()
