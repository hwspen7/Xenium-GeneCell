"""Read-only integration checks for SG-MAE datasets and variable panels."""

from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import torch
from scipy import sparse
from torch.utils.data import ConcatDataset, DataLoader

from src.data.gene_vocabulary import GeneVocabulary
from src.dataset import (
    collate_sgmae_batch,
    find_xenium_sample_file,
    initialize_sham_vocabulary,
    load_sgmae_sample,
    load_xenium_expression,
    load_xenium_gene_metadata,
)
from src.models.gene_aware_mae import GeneAwareMAE


def _write_mock_xenium(path, identifiers, symbols, counts):
    """Write minimal, representative Xenium feature-by-cell CSC H5."""
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = np.asarray(counts, dtype=np.float32)
    if counts.ndim != 2 or counts.shape[1] != len(identifiers):
        raise ValueError("Invalid synthetic count matrix.")

    features_by_cells = np.vstack([
        counts.T,
        np.zeros((1, counts.shape[0]), dtype=np.float32),
    ])
    csc = sparse.csc_matrix(features_by_cells)
    with h5py.File(path, "w") as handle:
        matrix = handle.create_group("matrix")
        matrix.create_dataset("shape", data=np.asarray(csc.shape, dtype=np.int64))
        matrix.create_dataset("data", data=csc.data)
        matrix.create_dataset("indices", data=csc.indices)
        matrix.create_dataset("indptr", data=csc.indptr)
        matrix.create_dataset("barcodes", data=np.asarray(
            [f"cell-{i}".encode() for i in range(counts.shape[0])]
        ))
        features = matrix.create_group("features")
        features.create_dataset("id", data=np.asarray(
            [name.encode() for name in identifiers] + [b"Control_001"]
        ))
        features.create_dataset("name", data=np.asarray(
            [name.encode() for name in symbols] + [b"Control"]
        ))
        features.create_dataset("feature_type", data=np.asarray(
            [b"Gene Expression"] * len(identifiers) + [b"Negative Control Probe"]
        ))


def _test_synthetic():
    """Exercise vocabulary mapping, mixed panels, padding and target alignment."""
    with TemporaryDirectory() as directory:
        root = Path(directory)
        train_path = root / "data" / "ShamL-demo" / "cell_feature_matrix.h5"
        other_path = root / "data" / "ShamR-demo" / "cell_feature_matrix.h5"
        vocab_path = root / "settings" / "vocab.json"

        _write_mock_xenium(
            train_path,
            ["ENSMUSG00000000001", "ENSMUSG00000000002", "ENSMUSG00000000003"],
            ["Aqp1", "Aqp2", "Spp1"],
            [[0, 5, 2], [7, 0, 0], [4, 2, 1]],
        )
        _write_mock_xenium(
            other_path,
            ["ENSMUSG00000000003", "ENSMUSG00000000001", "ENSMUSG00000000004", "ENSMUSG00000000005"],
            ["Spp1", "Aqp1", "Col1a1", "Havcr1"],
            [[9, 0, 6, 1], [0, 8, 3, 0]],
        )

        expression, cell_ids, symbols = load_xenium_expression(train_path)
        identifiers, symbols_from_meta = load_xenium_gene_metadata(train_path)
        assert expression.shape == (3, 3)
        assert symbols == symbols_from_meta
        assert len(cell_ids) == 3
        assert expression.nnz == 6
        assert find_xenium_sample_file(root / "data", "ShamL") == train_path
        print("Xenium H5 feature filtering: PASS")

        vocabulary = initialize_sham_vocabulary(train_path, vocab_path)
        assert len(vocabulary) == 3
        assert initialize_sham_vocabulary(train_path, vocab_path).fingerprint == vocabulary.fingerprint
        print("Explicit vocabulary initialization: PASS")

        try:
            load_sgmae_sample(other_path, vocabulary)
        except KeyError:
            pass
        else:
            raise AssertionError("Unknown measured gene was silently accepted.")
        print("Unknown measured gene rejection: PASS")

        vocabulary.extend(["ENSMUSG00000000004", "ENSMUSG00000000005"])
        train = load_sgmae_sample(train_path, vocabulary)
        other = load_sgmae_sample(other_path, vocabulary)
        assert train.vocabulary_fingerprint == other.vocabulary_fingerprint
        assert train.gene_ids[0] == other.gene_ids[1]
        assert train.gene_ids[2] == other.gene_ids[0]
        print("Stable ID alignment across panels: PASS")

        mixed = ConcatDataset([train, other])
        loader = DataLoader(
            mixed,
            batch_size=len(mixed),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_sgmae_batch,
        )
        batch = next(iter(loader))
        assert batch["expression"].shape == (5, 4)
        assert batch["gene_ids"].shape == (5, 4)
        assert batch["measured_mask"].shape == (5, 4)
        assert (batch["gene_ids"][:3, 3] == -1).all().item()
        assert not batch["measured_mask"][:3, 3].any().item()
        assert (batch["expression"][:3, 3] == 0).all().item()
        assert batch["measured_mask"][0, 0].item()
        assert batch["expression"][0, 0].item() == 0.0
        assert batch["vocabulary_fingerprint"] == vocabulary.fingerprint
        print("Mixed-panel padding and measured zeros: PASS")

        inconsistent = dict(other[0])
        inconsistent["vocabulary_fingerprint"] = "incorrect-vocabulary"
        try:
            collate_sgmae_batch([train[0], inconsistent])
        except ValueError as error:
            assert "vocabulary" in str(error)
        else:
            raise AssertionError("Mismatched vocabularies were mixed in one batch.")
        print("Cross-vocabulary batch mixing rejected: PASS")

        torch.manual_seed(17)
        model = GeneAwareMAE(
            vocab_size=len(vocabulary),
            token_dim=128,
            cell_dim=128,
            num_latents=8,
            num_heads=4,
            dropout=0.0,
        ).eval()
        reconstruction_mask = torch.zeros_like(batch["measured_mask"])
        reconstruction_mask[0, 1] = True
        reconstruction_mask[3, 2] = True

        with torch.no_grad():
            outputs = model(
                expression=batch["expression"],
                gene_ids=batch["gene_ids"],
                measured_mask=batch["measured_mask"],
                reconstruction_mask=reconstruction_mask,
                target_gene_ids=batch["gene_ids"],
            )
        assert outputs["target_counts"].shape == batch["expression"].shape
        assert torch.equal(outputs["target_counts"], batch["expression"])
        assert torch.equal(outputs["target_measured_mask"], batch["measured_mask"])
        assert torch.equal(outputs["target_reconstruction_mask"], reconstruction_mask)
        assert torch.isfinite(outputs["cell_embeddings"]).all().item()
        print("SG-MAE dynamic-batch forward/alignment: PASS")

        # Decoder targets may be requested in a different order.
        target_ids = torch.stack([batch["gene_ids"][:, 1], batch["gene_ids"][:, 0]], dim=1)
        with torch.no_grad():
            reordered = model(
                batch["expression"], batch["gene_ids"],
                batch["measured_mask"], reconstruction_mask,
                target_gene_ids=target_ids,
            )
        assert torch.equal(reordered["target_counts"], batch["expression"][:, [1, 0]])
        assert torch.equal(reordered["target_reconstruction_mask"], reconstruction_mask[:, [1, 0]])
        print("Reordered decoder targets: PASS")

        # A read-only test must never create the official vocabulary.
        assert not (root / "configs" / "gene_vocabulary.json").exists()
        print("Official configuration isolation: PASS")


def _test_real_if_available():
    """Check real ShamL/ShamR, without creating/updating the project vocabulary."""
    root = Path(__file__).resolve().parents[1]
    data_root = root / "data"
    try:
        train_path = find_xenium_sample_file(data_root, "ShamL")
        heldout_path = find_xenium_sample_file(data_root, "ShamR")
    except (NotADirectoryError, ValueError):
        print("Real ShamL/ShamR integration: SKIPPED (data unavailable)")
        return

    identifiers, _ = load_xenium_gene_metadata(train_path)
    official_path = root / "configs" / "gene_vocabulary.json"
    if official_path.exists():
        vocabulary = GeneVocabulary.load(official_path)
        assert vocabulary.namespace == "ensembl_gene_id"
        assert vocabulary.species == "Mus musculus"
        assert set(identifiers).issubset(vocabulary.gene_to_id)
        print("Existing official vocabulary: READ-ONLY VERIFIED")
    else:
        # In-memory temporary vocabulary; no file is written.
        vocabulary = GeneVocabulary.from_genes(
            identifiers, species="Mus musculus", namespace="ensembl_gene_id"
        )
        print("Official vocabulary absent: in-memory validation only")

    train = load_sgmae_sample(train_path, vocabulary)
    heldout = load_sgmae_sample(heldout_path, vocabulary)
    assert train.vocabulary_fingerprint == heldout.vocabulary_fingerprint
    assert set(train.gene_ids.tolist()) == set(heldout.gene_ids.tolist())
    assert len(train) > 0 and len(heldout) > 0
    assert len(train.gene_ids) == len(identifiers)

    loader = DataLoader(
        train, batch_size=4, shuffle=False, num_workers=0,
        collate_fn=collate_sgmae_batch,
    )
    batch = next(iter(loader))
    model = GeneAwareMAE(
        vocab_size=len(vocabulary), token_dim=128,
        cell_dim=128, num_latents=8, num_heads=4, dropout=0.0,
    ).eval()
    masks = torch.zeros_like(batch["measured_mask"])
    masks[:, ::5] = True
    with torch.no_grad():
        output = model(
            batch["expression"], batch["gene_ids"],
            batch["measured_mask"], masks,
        )
    assert torch.equal(output["target_counts"], batch["expression"])
    assert torch.equal(output["target_reconstruction_mask"], masks)
    assert torch.isfinite(output["cell_embeddings"]).all().item()
    print(f"Real ShamL / ShamR: PASS ({len(train):,} / {len(heldout):,} cells)")


def main():
    _test_synthetic()
    _test_real_if_available()
    print("\nAll SG-MAE dataset integration tests passed.")


if __name__ == "__main__":
    main()
