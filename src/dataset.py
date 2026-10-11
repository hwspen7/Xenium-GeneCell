"""Xenium H5 readers and gene-ID-aware datasets for SG-MAE V1.0.

The model receives raw Xenium counts. No normalization, masking, or
training/test split is performed in this module.
"""

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset

from src.data.gene_vocabulary import GeneVocabulary


_GENE_FEATURE_TYPE = "Gene Expression"


def decode_strings(values) -> list[str]:
    """Convert H5 byte or Unicode strings to Python strings."""
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def _read_xenium_h5(file_path: str | Path):
    """Read counts and aligned Ensembl IDs/symbols in one H5 pass."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Xenium H5 not found: {path}")

    with h5py.File(path, "r") as handle:
        if "matrix" not in handle:
            raise ValueError("Xenium H5 is missing the 'matrix' group.")
        matrix = handle["matrix"]
        required_matrix = {"shape", "data", "indices", "indptr", "barcodes", "features"}
        if not required_matrix.issubset(matrix.keys()):
            raise ValueError("Xenium H5 matrix is missing required fields.")

        features = matrix["features"]
        if not {"id", "name", "feature_type"}.issubset(features.keys()):
            raise ValueError("Xenium H5 feature metadata is incomplete.")

        shape = tuple(int(value) for value in matrix["shape"][:])
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError(f"Invalid Xenium matrix shape: {shape}")

        data = matrix["data"][:]
        indices = matrix["indices"][:]
        indptr = matrix["indptr"][:]
        cell_ids = decode_strings(matrix["barcodes"][:])
        feature_types = decode_strings(features["feature_type"][:])
        feature_ids = decode_strings(features["id"][:])
        feature_symbols = decode_strings(features["name"][:])

    if len(cell_ids) != shape[1] or len(set(cell_ids)) != len(cell_ids):
        raise ValueError("Cell IDs are duplicated or inconsistent with matrix shape.")
    if any(not cell_id or cell_id != cell_id.strip() for cell_id in cell_ids):
        raise ValueError("Invalid cell ID in H5.")
    if not (len(feature_types) == len(feature_ids) == len(feature_symbols) == shape[0]):
        raise ValueError("Feature metadata lengths disagree with matrix shape.")
    if not np.isfinite(data).all() or (data < 0).any():
        raise ValueError("Raw Xenium counts must be finite and nonnegative.")

    selected = np.flatnonzero(np.asarray(feature_types) == _GENE_FEATURE_TYPE)
    if selected.size == 0:
        raise ValueError("No 'Gene Expression' features found in H5.")

    gene_identifiers = [feature_ids[int(i)] for i in selected]
    gene_symbols = [feature_symbols[int(i)] for i in selected]
    if any(not gene_id or gene_id != gene_id.strip() for gene_id in gene_identifiers):
        raise ValueError("Invalid gene identifier in Xenium features/id.")
    if len(set(gene_identifiers)) != len(gene_identifiers):
        raise ValueError("Duplicate stable gene identifiers in H5.")

    matrix_csc = sparse.csc_matrix((data, indices, indptr), shape=shape)
    expression = matrix_csc[selected, :].T.tocsr().astype(np.float32)
    expression.sum_duplicates()
    expression.eliminate_zeros()
    expression.sort_indices()

    if not np.isfinite(expression.data).all() or (expression.data < 0).any():
        raise ValueError("Processed gene counts must be finite and nonnegative.")

    return expression, cell_ids, gene_identifiers, gene_symbols


def load_xenium_expression(file_path, expected_genes: int | None = None):
    """Legacy reader: return (cell-by-gene CSR, cell IDs, gene symbols).

    The optional expected_genes check supports old 300-gene tests, while
    allowing future panels with a different number of measured genes.
    """
    expression, cell_ids, _, gene_symbols = _read_xenium_h5(file_path)
    if expected_genes is not None and expression.shape[1] != expected_genes:
        raise ValueError(
            f"Expected {expected_genes} genes; found {expression.shape[1]}."
        )
    return expression, cell_ids, gene_symbols


def load_xenium_gene_metadata(file_path):
    """Read feature metadata without loading the large expression matrix."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Xenium H5 not found: {path}")
    with h5py.File(path, "r") as handle:
        if "matrix" not in handle or "features" not in handle["matrix"]:
            raise ValueError("Xenium H5 feature metadata is missing.")
        features = handle["matrix"]["features"]
        if not {"id", "name", "feature_type"}.issubset(features.keys()):
            raise ValueError("Xenium H5 feature metadata is incomplete.")
        types = decode_strings(features["feature_type"][:])
        ids = decode_strings(features["id"][:])
        names = decode_strings(features["name"][:])
    if not (len(types) == len(ids) == len(names)):
        raise ValueError("Feature metadata lengths disagree.")
    selected = [i for i, t in enumerate(types) if t == _GENE_FEATURE_TYPE]
    gene_ids = [ids[i] for i in selected]
    symbols = [names[i] for i in selected]
    if not gene_ids or len(set(gene_ids)) != len(gene_ids):
        raise ValueError("No genes or duplicate gene identifiers.")
    if any(not identifier or identifier != identifier.strip() for identifier in gene_ids):
        raise ValueError("Invalid gene identifier.")
    return gene_ids, symbols


def find_xenium_sample_file(data_root, sample_prefix: str) -> Path:
    """Require exactly one Xenium H5 for a named sample prefix."""
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data directory not found: {data_root}")
    if not isinstance(sample_prefix, str) or not sample_prefix.strip():
        raise ValueError("sample_prefix must be a non-empty string.")

    matches = sorted(
        path
        for path in data_root.rglob("cell_feature_matrix.h5")
        if path.parent.name == sample_prefix
        or path.parent.name.startswith(sample_prefix + "-")
    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected one H5 for sample '{sample_prefix}', found {len(matches)}."
        )
    return matches[0]


def initialize_sham_vocabulary(
    sham_h5_path,
    vocabulary_path,
) -> GeneVocabulary:
    """Explicitly create a Sham-only vocabulary or verify an existing one.

    Existing files are never overwritten or silently expanded.
    Call from the initialization CLI, not from read-only tests.
    """
    vocabulary_path = Path(vocabulary_path)
    identifiers, _ = load_xenium_gene_metadata(sham_h5_path)

    if vocabulary_path.exists():
        vocabulary = GeneVocabulary.load(vocabulary_path)
        if (
            vocabulary.species != "Mus musculus"
            or vocabulary.namespace != "ensembl_gene_id"
        ):
            raise ValueError("Existing vocabulary metadata is incompatible.")
        missing = set(identifiers).difference(vocabulary.gene_to_id)
        if missing:
            raise ValueError(
                f"Existing vocabulary lacks {len(missing)} ShamL gene IDs; "
                "review before explicitly extending it."
            )
        return vocabulary

    vocabulary = GeneVocabulary.from_genes(
        identifiers,
        species="Mus musculus",
        namespace="ensembl_gene_id",
    )
    vocabulary.save(vocabulary_path)
    return vocabulary


class XeniumCellDataset(Dataset):
    """Legacy per-sample PyTorch Dataset yielding only raw count tensors."""

    def __init__(self, expression, cell_ids, gene_names, sample_name):
        if not sparse.issparse(expression):
            raise TypeError("Expression must be sparse.")
        if expression.shape != (len(cell_ids), len(gene_names)):
            raise ValueError("Expression and metadata dimensions disagree.")
        self.expression = expression.tocsr().astype(np.float32)
        self.cell_ids = list(cell_ids)
        self.gene_names = list(gene_names)
        self.sample_name = str(sample_name)

    def __len__(self):
        return self.expression.shape[0]

    def __getitem__(self, index):
        counts = self.expression.getrow(int(index)).toarray().ravel().copy()
        return torch.from_numpy(counts)


def load_xenium_samples(data_root):
    """Legacy four-sample loader, aligning gene symbols by name.

    New SG-MAE code should use load_sgmae_sample and Ensembl IDs instead.
    """
    files = sorted(Path(data_root).rglob("cell_feature_matrix.h5"))
    if len(files) != 4:
        raise ValueError(f"Expected four Xenium H5 files, found {len(files)}.")
    reference_genes = None
    datasets = {}

    for file_path in files:
        sample_name = file_path.parent.name
        if sample_name in datasets:
            raise ValueError(f"Repeated sample folder name: {sample_name}")
        expression, cell_ids, gene_names = load_xenium_expression(file_path)
        if len(set(gene_names)) != len(gene_names):
            raise ValueError(f"Legacy symbol alignment is ambiguous: {sample_name}")
        if reference_genes is None:
            reference_genes = sorted(gene_names)
        if set(gene_names) != set(reference_genes):
            raise ValueError(f"Gene panel mismatch: {sample_name}")
        gene_index = {name: i for i, name in enumerate(gene_names)}
        columns = [gene_index[name] for name in reference_genes]
        datasets[sample_name] = XeniumCellDataset(
            expression[:, columns].tocsr(), cell_ids, reference_genes, sample_name
        )

    return datasets, reference_genes


class SGMAECellDataset(Dataset):
    """One sample of raw counts indexed by persistent global gene IDs.

    A single sample's H5 columns represent measured genes. Therefore its
    per-item measured_mask is all True. Padding and unmeasured genes are
    introduced only when the mixed-panel collator assembles a batch.
    """

    def __init__(
        self,
        expression: sparse.spmatrix,
        cell_ids: Sequence[str],
        gene_identifiers: Sequence[str],
        gene_symbols: Sequence[str],
        vocabulary: GeneVocabulary,
        sample_name: str,
    ):
        if not sparse.issparse(expression):
            raise TypeError("expression must be a SciPy sparse matrix.")
        if not isinstance(vocabulary, GeneVocabulary):
            raise TypeError("vocabulary must be GeneVocabulary.")
        if vocabulary.species != "Mus musculus" or vocabulary.namespace != "ensembl_gene_id":
            raise ValueError("Expected Mus musculus / ensembl_gene_id vocabulary.")

        num_cells, num_genes = expression.shape
        if num_cells < 1 or num_genes < 1:
            raise ValueError("Empty cell or gene dimension.")
        if (
            len(cell_ids) != num_cells
            or len(gene_identifiers) != num_genes
            or len(gene_symbols) != num_genes
        ):
            raise ValueError("Count matrix and metadata dimensions disagree.")
        if len(set(cell_ids)) != num_cells or len(set(gene_identifiers)) != num_genes:
            raise ValueError("Duplicate cell ID or measured gene ID.")
        if any(not isinstance(gene, str) or not gene for gene in gene_identifiers):
            raise ValueError("Invalid stable gene identifier.")

        matrix = expression.tocsr().astype(np.float32)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
            raise ValueError("Counts must be finite and nonnegative.")

        # Unknown measured genes must never be silently treated as padding.
        global_ids = vocabulary.encode(list(gene_identifiers), allow_unknown=False)
        self.expression = matrix
        self.cell_ids = list(cell_ids)
        self.gene_identifiers = list(gene_identifiers)
        self.gene_symbols = list(gene_symbols)
        self.gene_ids = torch.tensor(global_ids, dtype=torch.long)
        self.measured_mask = torch.ones(num_genes, dtype=torch.bool)
        self.sample_name = str(sample_name)
        self.vocabulary_fingerprint = vocabulary.fingerprint

    def __len__(self):
        return self.expression.shape[0]

    def __getitem__(self, index):
        idx = int(index)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Cell row outside dataset: {idx}")
        counts = self.expression.getrow(idx).toarray().ravel().copy()
        return {
            "expression": torch.from_numpy(counts),
            "gene_ids": self.gene_ids,
            "measured_mask": self.measured_mask,
            "vocabulary_fingerprint": self.vocabulary_fingerprint,
        }


def load_sgmae_sample(file_path, vocabulary: GeneVocabulary) -> SGMAECellDataset:
    """Read Xenium H5 once and construct a gene-ID-aware sample Dataset."""
    expression, cell_ids, identifiers, symbols = _read_xenium_h5(file_path)
    return SGMAECellDataset(
        expression=expression,
        cell_ids=cell_ids,
        gene_identifiers=identifiers,
        gene_symbols=symbols,
        vocabulary=vocabulary,
        sample_name=Path(file_path).parent.name,
    )


def collate_sgmae_batch(batch):
    """Pad heterogeneous gene panels in a batch to [B, max_G].

    Pad expression with 0, gene_ids with -1, and measured_mask with False.
    Existing unmeasured positions remain unmeasured. This function does not
    construct reconstruction masks or consult expression to select targets.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    required = {"expression", "gene_ids", "measured_mask"}
    fingerprints = [item.get("vocabulary_fingerprint") if isinstance(item, dict) else None for item in batch]
    if any(fingerprint is not None for fingerprint in fingerprints):
        if (
            any(not isinstance(fingerprint, str) or not fingerprint for fingerprint in fingerprints)
            or len(set(fingerprints)) != 1
        ):
            raise ValueError("Batch contains different or missing vocabulary fingerprints.")
    widths = []
    for item in batch:
        if not isinstance(item, dict) or not required.issubset(item):
            raise TypeError("Each sample must contain expression, gene_ids, measured_mask.")
        expression, ids, measured = (item[key] for key in ("expression", "gene_ids", "measured_mask"))
        if any(not isinstance(tensor, torch.Tensor) for tensor in (expression, ids, measured)):
            raise TypeError("Collated values must be torch tensors.")
        if expression.ndim != 1 or ids.ndim != 1 or measured.ndim != 1:
            raise ValueError("Expected one-dimensional per-cell gene vectors.")
        if expression.numel() < 1 or expression.shape != ids.shape or ids.shape != measured.shape:
            raise ValueError("Per-cell expression, ids, measured_mask shapes disagree.")
        if not expression.is_floating_point() or ids.dtype != torch.long or measured.dtype != torch.bool:
            raise TypeError("Expected float counts, torch.long IDs, and boolean mask.")
        if (ids < -1).any().item() or (measured & (ids == -1)).any().item():
            raise ValueError("Invalid padding or measured gene IDs.")
        valid_ids = ids[ids >= 0]
        if torch.unique(valid_ids).numel() != valid_ids.numel():
            raise ValueError("Duplicate non-padding gene IDs in cell.")
        valid_counts = expression[measured]
        if not torch.isfinite(valid_counts).all().item() or (valid_counts < 0).any().item():
            raise ValueError("Measured counts must be finite and nonnegative.")
        widths.append(expression.numel())

    max_width = max(widths)
    size = len(batch)
    result_expression = torch.zeros(size, max_width, dtype=torch.float32)
    result_ids = torch.full((size, max_width), -1, dtype=torch.long)
    result_measured = torch.zeros(size, max_width, dtype=torch.bool)

    for row, item in enumerate(batch):
        width = widths[row]
        measured = item["measured_mask"]
        result_expression[row, :width] = torch.where(
            measured,
            item["expression"],
            torch.zeros_like(item["expression"]),
        ).to(dtype=torch.float32)
        result_ids[row, :width] = item["gene_ids"]
        result_measured[row, :width] = measured

    output = {
        "expression": result_expression,
        "gene_ids": result_ids,
        "measured_mask": result_measured,
    }
    if fingerprints[0] is not None:
        output["vocabulary_fingerprint"] = fingerprints[0]
    return output
