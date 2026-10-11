"""Evaluate frozen SG-MAE models across four Xenium slices and plot results.

ShamL contributes only its frozen spatial validation cells. All cells in
ShamR, Hour12R and Day2L are external evaluation targets, never training
inputs. Checkpoints, masks, baselines and model selection stay unchanged.

Run from the project root:
    python -m scripts.evaluate_sgmae_conditions \
        --checkpoint baseline=runs/sgmae_v1_baseline/best.pt \
        --checkpoint scheduler=runs/sgmae_v1_scheduler/best.pt \
        --confirm-external \
        --output-dir runs/sgmae_v1_four_slices
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.evaluate_sgmae_v1 import main as evaluate_one_sample


SAMPLES = (
    ("shaml", "ShamL", "Control", "Validation"),
    ("shamr", "ShamR", "Control", "External slice"),
    ("hour12r", "12h", "Injury", "External slice"),
    ("day2l", "Day 2", "Injury", "External slice"),
)

METRICS = (
    ("reconstruction_loss", "Reconstruction loss", False),
    ("presence_auprc", "Presence AUPRC", True),
    ("presence_auroc", "Presence AUROC", True),
    ("presence_f1", "Presence F1 @ 0.5", True),
    ("positive_mae", "Positive log1p MAE", False),
)


def _parse_checkpoint(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("Use --checkpoint name=path/to/best.pt")
    label, path = text.split("=", 1)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,39}", label):
        raise argparse.ArgumentTypeError("Invalid checkpoint label")
    if Path(path).name != "best.pt":
        raise argparse.ArgumentTypeError("Only best.pt checkpoints are supported")
    return label, path


def _metric_values(result: dict, method: str) -> dict:
    values = result["model"] if method == "model" else result["baseline"]
    loss = result["reconstruction_loss"]
    return {
        "reconstruction_loss": loss["total"] if method == "model" else loss["baseline_total"],
        "presence_auprc": values["presence"]["auprc"],
        "presence_auroc": values["presence"]["auroc"],
        "presence_f1": values["presence"]["f1_at_0_5"],
        "positive_mae": values["positive_intensity_log1p"]["mae"],
        "presence_macro_gene_auprc": values["presence"]["macro_gene_auprc"],
        "presence_macro_gene_auroc": values["presence"]["macro_gene_auroc"],
    }


def _ensure_comparable(reports: dict, labels: list[str]) -> None:
    """Ensure every row represents the same preselected frozen experiment."""
    first = reports["shaml"]
    common_fields = (
        "vocabulary_fingerprint",
        "split_fingerprint",
        "shaml_h5_sha256",
        "fixed_protocol",
        "num_shaml_train_cells_for_baseline",
    )
    for sample, report in reports.items():
        if report.get("sample") != sample:
            raise ValueError(f"Report mislabeled as {sample}.")
        for field in common_fields:
            if report.get(field) != first.get(field):
                raise ValueError(f"Cross-slice mismatch: {field} ({sample})")
        if set(report["results"]) != set(labels):
            raise ValueError(f"Checkpoint labels differ for {sample}.")
        expected_heldout = sample != "shaml"
        if report.get("heldout") != expected_heldout:
            raise ValueError(f"Incorrect held-out flag for {sample}.")
        for label in labels:
            current = report["results"][label]
            reference = first["results"][label]
            if current["checkpoint_sha256"] != reference["checkpoint_sha256"]:
                raise ValueError(f"Different checkpoint bytes for {label}.")
            if current["loss_config"] != reference["loss_config"]:
                raise ValueError(f"Different loss definition for {label}.")
            if current["masked_targets"] <= 0:
                raise ValueError(f"No evaluated targets in {sample}.")
        # Cross-model masks are equal inside each sample, not across samples.
        hashes = {report["results"][label]["mask_sha256"] for label in labels}
        if len(hashes) != 1:
            raise ValueError(f"Evaluation masks differ across checkpoints in {sample}.")


def _build_rows(reports: dict, labels: list[str]) -> list[dict]:
    records = []
    for key, display, condition, scope in SAMPLES:
        report = reports[key]
        first = report["results"][labels[0]]
        rows_to_add = [("Gene-frequency baseline", "baseline", first)]
        rows_to_add += [(label, "model", report["results"][label]) for label in labels]
        for model_name, kind, result in rows_to_add:
            metrics = _metric_values(result, kind)
            records.append({
                "sample": key,
                "display": display,
                "condition": condition,
                "scope": scope,
                "method": model_name,
                "cells": result["num_cells"],
                "masked_targets": result["masked_targets"],
                "positive_targets": result["positive_targets"],
                "positive_prevalence": result["model"]["presence"]["positive_prevalence"],
                **metrics,
            })
    return records


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot save an empty comparison.")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict], labels: list[str], destination: Path) -> None:
    """Create one black-and-white multi-panel comparison for all four slices."""
    methods = ["Gene-frequency baseline", *labels]
    sample_ids = [item[0] for item in SAMPLES]
    sample_names = [item[1] for item in SAMPLES]
    lookup = {(row["sample"], row["method"]): row for row in rows}
    x = np.arange(len(sample_ids), dtype=float)
    width = 0.78 / len(methods)
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width
    shades = ["#d7d7d7", "#888888", "#222222", "#555555"]
    hatch_patterns = ["//", "", "..", "xx"]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.3), constrained_layout=True)
    for ax, (metric, title, higher_is_better) in zip(axes.flat[:5], METRICS):
        for i, method in enumerate(methods):
            heights = []
            for sample in sample_ids:
                value = lookup[(sample, method)][metric]
                heights.append(float(value) if value is not None else float("nan"))
            ax.bar(
                x + offsets[i], heights, width=width * 0.96,
                label=method, color=shades[i], edgecolor="black",
                linewidth=0.5, hatch=hatch_patterns[i],
            )
        ax.set_title(f"{title} ({'higher' if higher_is_better else 'lower'} is better)")
        ax.set_xticks(x, sample_names)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.set_axisbelow(True)
        if metric in ("presence_auprc", "presence_auroc", "presence_f1"):
            ax.set_ylim(0, 1.02)
        else:
            ax.set_ylim(bottom=0)
        ax.axvline(1.5, color="#777777", linestyle="--", linewidth=0.8)

    ax = axes.flat[5]
    prevalences = [lookup[(sample, labels[0])]["positive_prevalence"] for sample in sample_ids]
    ax.bar(x, prevalences, width=0.55, color="#888888", edgecolor="black")
    ax.set_title("Positive target prevalence (sample context)")
    ax.set_xticks(x, sample_names)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Fraction of masked genes with nonzero counts")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)
    ax.axvline(1.5, color="#777777", linestyle="--", linewidth=0.8)

    handles, legends = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legends, loc="upper center", ncol=len(methods),
               bbox_to_anchor=(0.5, 1.035), frameon=False)
    fig.suptitle("SG-MAE V1.0 | Zero-shot masked reconstruction across kidney slices",
                 y=1.073, fontsize=15, fontweight="bold")
    fig.text(0.5, -0.025,
             "ShamL: spatial validation. ShamR, 12h and Day 2: external slices. "
             "Only ShamL training cells fit the frequency baseline. "
             "One slice per injury timepoint; results do not establish disease prediction.",
             ha="center", va="top", fontsize=9)
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Four-slice, frozen SG-MAE V1 evaluation and comparison figure"
    )
    parser.add_argument("--checkpoint", action="append", required=True, type=_parse_checkpoint)
    parser.add_argument("--data-root", type=Path, default=root / "data")
    parser.add_argument("--vocabulary", type=Path, default=root / "configs" / "gene_vocabulary.json")
    parser.add_argument("--split", type=Path, default=root / "configs" / "splits" / "shaml_spatial_v1.json.gz")
    parser.add_argument("--output-dir", type=Path, default=root / "runs" / "sgmae_v1_four_slices")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--confirm-external", action="store_true")
    args = parser.parse_args(argv)
    labels = [label for label, _ in args.checkpoint]
    if len(set(labels)) != len(labels):
        parser.error("Checkpoint labels must be unique.")
    if not args.confirm_external:
        parser.error("Use --confirm-external to authorize evaluation of three external slices.")
    if args.output_dir.exists():
        raise FileExistsError(
            f"Output folder already exists: {args.output_dir}. "
            "Use a new output folder; reports are never overwritten."
        )

    common = [
        "--data-root", str(args.data_root),
        "--vocabulary", str(args.vocabulary),
        "--split", str(args.split),
        "--device", args.device,
    ]
    for label, path in args.checkpoint:
        common.extend(("--checkpoint", f"{label}={path}"))

    reports = {}
    for sample, display, _, _ in SAMPLES:
        print(f"\n{'=' * 60}\nEvaluating {display} ({sample})\n{'=' * 60}")
        arguments = common + [
            "--sample", sample,
            "--output-dir", str(args.output_dir / sample),
        ]
        if sample != "shaml":
            arguments.append("--confirm-heldout")
        reports[sample] = evaluate_one_sample(arguments)

    _ensure_comparable(reports, labels)
    rows = _build_rows(reports, labels)
    _write_csv(args.output_dir / "comparison.csv", rows)
    figure_path = args.output_dir / "four_slice_comparison.png"
    _plot(rows, labels, figure_path)
    manifest = {
        "samples": [item[0] for item in SAMPLES],
        "checkpoints": labels,
        "vocabulary_fingerprint": reports["shaml"]["vocabulary_fingerprint"],
        "split_fingerprint": reports["shaml"]["split_fingerprint"],
        "selection_policy": "Select the preferred model on ShamL only; never optimize using external slices.",
        "notes": "One slice per injury timepoint: differences may be confounded with biological or technical variation.",
    }
    with (args.output_dir / "comparison_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(f"\nCombined CSV: {args.output_dir / 'comparison.csv'}")
    print(f"Combined figure: {figure_path}")
    print("Frozen checkpoints and splits remain unchanged.")
    return rows


if __name__ == "__main__":
    main()
