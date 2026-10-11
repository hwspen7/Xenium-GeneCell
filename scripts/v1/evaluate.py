"""Evaluate frozen SG-MAE checkpoints on ShamL or external Xenium slices.

Usage (from the project root):
  python -m scripts.v1.evaluate --checkpoint baseline=runs/sgmae_v1_baseline/best.pt --checkpoint scheduler=runs/sgmae_v1_scheduler/best.pt

External ShamR, Hour12R and Day2L evaluations require --confirm-heldout.
Never tune checkpoints or thresholds using these external slices.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

from src.data.gene_vocabulary import GeneVocabulary
from src.data.splitting import load_spatial_split
from src.dataset import find_xenium_sample_file, load_sgmae_sample
from src.evaluation.evaluate_sgmae_v1 import (
    EVALUATION_SCHEMA_VERSION,
    evaluate_frozen_reconstruction,
    fit_train_only_baseline,
    load_frozen_best,
    sha256_file,
)


def _parse_checkpoint(text: str) -> tuple[str, Path]:
    """Require explicit names so report filenames cannot collide."""
    if "=" not in text:
        raise argparse.ArgumentTypeError("Use --checkpoint name=path/to/best.pt")
    name, path = text.split("=", 1)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,39}", name) or not path:
        raise argparse.ArgumentTypeError("Invalid checkpoint label or path.")
    if Path(path).name != "best.pt":
        raise argparse.ArgumentTypeError("Only a best.pt checkpoint may be evaluated.")
    return name, Path(path)


def _write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        raise ValueError("Cannot write an empty per-gene report.")
    with path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main(argv=None):
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Frozen SG-MAE V1 reconstruction evaluation")
    parser.add_argument("--checkpoint", action="append", type=_parse_checkpoint, required=True)
    parser.add_argument("--data-root", type=Path, default=root / "data")
    parser.add_argument("--vocabulary", type=Path, default=root / "configs" / "gene_vocabulary.json")
    parser.add_argument("--split", type=Path, default=root / "configs" / "splits" / "shaml_spatial_v1.json.gz")
    parser.add_argument("--sample", choices=("shaml", "shamr", "hour12r", "day2l"), default="shaml")
    parser.add_argument("--confirm-heldout", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scatter-size", type=int, default=6000, help="Deterministic visualization samples per checkpoint (0 disables)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)

    if args.scatter_size < 0:
        parser.error("--scatter-size must be nonnegative.")
    checkpoints = args.checkpoint
    labels = [label for label, _ in checkpoints]
    if len(set(labels)) != len(labels):
        parser.error("Checkpoint labels must be unique.")
    if args.sample != "shaml" and not args.confirm_heldout:
        parser.error("External slices require --confirm-heldout and a frozen evaluation protocol.")
    if args.sample == "shaml" and args.confirm_heldout:
        parser.error("--confirm-heldout is only valid for external slices.")
    output_dir = args.output_dir or root / "runs" / "v1" / "evaluation" / f"{args.sample}_new"
    if output_dir.exists():
        raise FileExistsError(
            f"Report directory already exists: {output_dir}. "
            "Use a new output directory instead of overwriting prior evaluation reports."
        )
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    vocabulary = GeneVocabulary.load(args.vocabulary)
    if vocabulary.species != "Mus musculus" or vocabulary.namespace != "ensembl_gene_id":
        raise ValueError("Evaluation expects mouse Ensembl Gene IDs.")

    sham_l_path = find_xenium_sample_file(args.data_root, "ShamL")
    sham_l = load_sgmae_sample(sham_l_path, vocabulary)
    split = load_spatial_split(
        args.split,
        sham_l.cell_ids,
        sham_l_path.parent / "cells.parquet",
        expected_sample_name="ShamL",
    )
    sham_l_sha = sha256_file(sham_l_path)
    baseline = fit_train_only_baseline(sham_l, split.train_indices)
    sample_prefix = {
        "shamr": "ShamR",
        "hour12r": "Hour12R",
        "day2l": "Day2L",
    }
    if args.sample == "shaml":
        evaluated = sham_l
        indices = split.validation_indices
        evaluated_h5 = sham_l_path
    else:
        evaluated_h5 = find_xenium_sample_file(
            args.data_root, sample_prefix[args.sample]
        )
        evaluated = load_sgmae_sample(evaluated_h5, vocabulary)
        indices = np.arange(len(evaluated), dtype=np.int64)

    # A common gene panel is required for interpretable cross-condition scores.
    if set(evaluated.gene_ids.tolist()) != set(sham_l.gene_ids.tolist()):
        raise ValueError(
            "The evaluation sample does not share ShamL's exact measured "
            "gene panel. Cross-panel comparison needs a separate protocol."
        )
    evaluation_h5_sha = sha256_file(evaluated_h5)
    gene_symbols_by_id = {
        int(gene_id): symbol
        for gene_id, symbol in zip(evaluated.gene_ids.tolist(), evaluated.gene_symbols)
    }

    print("SG-MAE V1.0 | Read-only reconstruction evaluation")
    display_name = {
        "shaml": "ShamL spatial validation",
        "shamr": "ShamR external control",
        "hour12r": "Hour12R injury 12h",
        "day2l": "Day2L injury day 2",
    }
    print(f"Evaluation: {display_name[args.sample]}")
    print(f"Cells: {len(indices):,} | device: {device}")
    print(f"Vocabulary: {vocabulary.fingerprint}")
    print(f"Frozen spatial split: {split.fingerprint}")
    print(f"ShamL training H5 SHA-256: {sham_l_sha}")

    reports = {}
    per_gene_reports = {}
    scatter_reports = {}
    shared_protocol = None
    shared_mask_hash = None
    shared_model_config = None
    shared_loss_config = None
    for label, path in checkpoints:
        print(f"\nEvaluating {label}: {path}")
        model, checkpoint, metadata = load_frozen_best(
            path,
            vocabulary_fingerprint=vocabulary.fingerprint,
            split_fingerprint=split.fingerprint,
            sham_h5_sha256=sham_l_sha,
            vocabulary_size=len(vocabulary),
            device=device,
        )
        settings = metadata["trainer_config"]
        if shared_model_config is None:
            shared_model_config = metadata["model_config"]
            shared_loss_config = metadata["loss_config"]
        elif (
            metadata["model_config"] != shared_model_config
            or metadata["loss_config"] != shared_loss_config
        ):
            raise ValueError("Checkpoints use different model or loss definitions.")
        protocol = {
            "batch_size": settings["batch_size"],
            "mask_ratio": settings["mask_ratio"],
            "validation_mask_seed": settings["validation_mask_seed"],
            "training_seed": settings.get("seed"),
        }
        if shared_protocol is None:
            shared_protocol = protocol
        elif protocol != shared_protocol:
            raise ValueError("Checkpoints have different mask, batch, or validation-seed protocols.")
        metrics, per_gene = evaluate_frozen_reconstruction(
            model=model,
            dataset=evaluated,
            cell_indices=indices,
            baseline=baseline,
            config=settings,
            loss_config=metadata["loss_config"],
            device=device,
            scatter_sample_size=args.scatter_size,
        )
        scatter_data = metrics.pop("plot_samples", None)
        if scatter_data is not None:
            scatter_reports[label] = scatter_data
        if shared_mask_hash is None:
            shared_mask_hash = metrics["mask_sha256"]
        elif shared_mask_hash != metrics["mask_sha256"]:
            raise AssertionError("Evaluations used different reconstruction masks.")
        if args.sample == "shaml":
            expected_loss = float(checkpoint["best_validation"])
            delta = abs(metrics["reconstruction_loss"]["total"] - expected_loss)
            if delta > 5e-4:
                raise ValueError(
                    f"Reproduced validation loss differs from saved best: "
                    f"computed={metrics['reconstruction_loss']['total']:.6f}, "
                    f"checkpoint={expected_loss:.6f}. Check data order and masking."
                )
            saved_history = checkpoint.get("history", [])
            if saved_history:
                last_val = saved_history[-1]["validation"]
                if (
                    int(last_val["num_masked"]) != metrics["masked_targets"]
                    or int(last_val["num_positive"]) != metrics["positive_targets"]
                ):
                    raise ValueError("Recomputed validation target counts differ from checkpoint.")
        for row in per_gene:
            row["ensembl_gene_id"] = vocabulary.decode([row["gene_id"]])[0]
            row["gene_symbol"] = gene_symbols_by_id.get(row["gene_id"], "")
        reports[label] = {
            "checkpoint": str(path),
            "checkpoint_sha256": sha256_file(path),
            "best_epoch": int(checkpoint["best_epoch"]),
            "checkpoint_best_validation_loss": float(checkpoint["best_validation"]),
            "loss_config": metadata["loss_config"],
            **metrics,
        }
        per_gene_reports[label] = per_gene
        print(f"  Masked targets: {metrics['masked_targets']:,} | positive: {metrics['positive_targets']:,}")
        print(f"  Loss: {metrics['reconstruction_loss']['total']:.6f} | baseline: {metrics['reconstruction_loss']['baseline_total']:.6f}")
        for key in ("auprc", "auroc", "f1_at_0_5"):
            print(f"  Presence {key}: {metrics['model']['presence'][key]} | baseline: {metrics['baseline']['presence'][key]}")
        print(f"  Positive log1p MAE: {metrics['model']['positive_intensity_log1p']['mae']} | baseline: {metrics['baseline']['positive_intensity_log1p']['mae']}")
        print(f"  Positive macro Pearson: {metrics['model']['positive_intensity_log1p']['macro_gene_pearson']}")
        print(f"  Positive macro Spearman: {metrics['model']['positive_intensity_log1p']['macro_gene_spearman']}")
        print(f"  Full macro Pearson: {metrics['model']['full_reconstruction_log1p']['macro_gene_pearson']}")
        print(f"  Full macro Spearman: {metrics['model']['full_reconstruction_log1p']['macro_gene_spearman']}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    report = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "sample": args.sample,
        "heldout": args.sample != "shaml",
        "condition": "sham" if args.sample in ("shaml", "shamr") else "injury",
        "timepoint": {"shaml": "control", "shamr": "control", "hour12r": "12h", "day2l": "day2"}[args.sample],
        "evaluation_scope": "spatial_validation" if args.sample == "shaml" else "external_slice",
        "vocabulary_fingerprint": vocabulary.fingerprint,
        "split_fingerprint": split.fingerprint,
        "shaml_h5_sha256": sham_l_sha,
        "evaluated_sample_h5_sha256": evaluation_h5_sha,
        "num_shaml_train_cells_for_baseline": baseline["num_train_cells"],
        "fixed_protocol": shared_protocol,
        "shared_mask_sha256": shared_mask_hash,
        "selection_policy": "Checkpoint selection by ShamL validation only. Presence threshold pre-fixed at 0.5.",
        "results": reports,
    }
    # Write reports only after every checkpoint passes provenance and protocol checks.
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    for label, rows in per_gene_reports.items():
        _write_csv(output_dir / f"{label}_per_gene.csv", rows)
    for label, data in scatter_reports.items():
        np.savez_compressed(
            output_dir / f"{label}_scatter_samples.npz",
            **{field: np.asarray(values) for field, values in data.items()},
        )
    print(f"\nReports saved: {output_dir}")
    print("No training files, checkpoints, or frozen split manifests were modified.")
    return report


if __name__ == "__main__":
    main()
