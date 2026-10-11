"""Regression checks for SG-MAE dynamic masking and model integration."""

import torch
from scipy import sparse
from torch.utils.data import ConcatDataset, DataLoader

from src.data.gene_vocabulary import GeneVocabulary
from src.data.masking import (
    generate_random_mask,
    generate_reconstruction_mask,
)
from src.dataset import SGMAECellDataset, collate_sgmae_batch
from src.models.gene_aware_mae import GeneAwareMAE


def expect_error(call, error_type):
    try:
        call()
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__}.")


def test_basic_masking():
    measured = torch.ones(8, 300, dtype=torch.bool)
    generator = torch.Generator(device="cpu").manual_seed(7)
    mask = generate_reconstruction_mask(measured, 0.2, generator=generator)
    assert mask.shape == measured.shape and mask.dtype == torch.bool
    assert torch.equal(mask.sum(1), torch.full((8,), 60))
    assert (mask & ~measured).sum().item() == 0
    print("Measured-only 300-gene masking: PASS")


def test_dynamic_panels():
    measured = torch.tensor([
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
    ], dtype=torch.bool)
    gene_ids = torch.tensor([
        [5, 9, 20, -1, -1, -1, -1],
        [2, 7, -1, 15, 40, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -1, 6, -1, -1, -1, -1],
    ])
    mask = generate_reconstruction_mask(measured, 0.5, gene_ids=gene_ids)
    assert mask.sum(1).tolist() == [2, 2, 0, 1]
    assert not (mask & ~measured).any().item()
    assert mask[3, 2].item()
    print("Mixed panels, padding and empty measurements: PASS")


def test_measured_zero_and_no_leakage():
    expression = torch.tensor([[0., 5., 0., 9., 2.]])
    measured = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    # The last two entries are unmeasured, regardless of their count values.
    first = generate_random_mask(
        expression, 0.4, measured_mask=measured,
        generator=torch.Generator().manual_seed(91),
    )
    changed = expression.clone()
    changed[measured] = torch.tensor([99999., 0., 1234.])
    second = generate_random_mask(
        changed, 0.4, measured_mask=measured,
        generator=torch.Generator().manual_seed(91),
    )
    assert torch.equal(first, second)
    assert first.sum().item() == 1
    assert not first[0, 3:].any().item()
    print("Measured zeros and count-independent selection: PASS")


def test_repeatability():
    measured = torch.ones(16, 100, dtype=torch.bool)
    first = generate_reconstruction_mask(
        measured, generator=torch.Generator().manual_seed(42)
    )
    second = generate_reconstruction_mask(
        measured, generator=torch.Generator().manual_seed(42)
    )
    assert torch.equal(first, second)
    generator = torch.Generator().manual_seed(42)
    a = generate_reconstruction_mask(measured, generator=generator)
    b = generate_reconstruction_mask(measured, generator=generator)
    assert not torch.equal(a, b)
    print("Seed reproducibility and independent train draws: PASS")


def test_mask_fraction_edges():
    measured = torch.tensor([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
    ], dtype=torch.bool)
    assert not generate_reconstruction_mask(measured, 0.0).any().item()
    full = generate_reconstruction_mask(measured, 1.0)
    assert full.sum(1).tolist() == [1, 1, 3, 0]
    allow_full = generate_reconstruction_mask(measured, 1.0, ensure_visible=False)
    assert allow_full.sum(1).tolist() == [1, 2, 4, 0]
    small = generate_reconstruction_mask(measured, 0.001)
    assert small.sum(1).tolist() == [1, 1, 1, 0]
    print("Zero ratio, all-hidden option and small panels: PASS")


def test_invalid_inputs():
    measured = torch.ones(2, 4, dtype=torch.bool)
    for ratio in (-0.1, 1.1, float("nan"), float("inf")):
        expect_error(lambda r=ratio: generate_reconstruction_mask(measured, r), ValueError)
    expect_error(lambda: generate_reconstruction_mask(measured, "0.2"), TypeError)
    expect_error(lambda: generate_reconstruction_mask(measured.float()), TypeError)
    expect_error(lambda: generate_reconstruction_mask(torch.ones(4, dtype=torch.bool)), ValueError)
    expect_error(lambda: generate_reconstruction_mask(measured, gene_ids=torch.tensor([1, 1, 2, 3])), ValueError)
    expect_error(lambda: generate_reconstruction_mask(measured, gene_ids=torch.tensor([1, -1, 2, 3])), ValueError)
    expect_error(lambda: generate_random_mask(torch.ones(2, 3), measured_mask=measured), ValueError)
    print("Input validation and duplicate-ID rejection: PASS")


def test_old_api():
    expression = torch.ones(3, 300)
    mask = generate_random_mask(expression, mask_ratio=0.2)
    assert mask.sum(dim=1).tolist() == [60, 60, 60]
    print("Legacy masking API compatibility: PASS")


def test_large_panel():
    measured = torch.ones(3, 5000, dtype=torch.bool)
    measured[1, 1000:] = False
    measured[2, 2500:] = False
    ids = torch.arange(5000)
    mask = generate_reconstruction_mask(measured, gene_ids=ids)
    assert mask.sum(1).tolist() == [1000, 200, 500]
    assert not (mask & ~measured).any().item()
    print("5000-gene dynamic-panel masking: PASS")


def test_model_and_dataset_integration():
    vocab = GeneVocabulary.from_genes(
        [f"ENSMUSG{i:011d}" for i in range(6)],
        namespace="ensembl_gene_id",
    )
    a = SGMAECellDataset(
        expression=sparse.csr_matrix([[0., 4., 2.], [3., 0., 0.]]),
        cell_ids=["a1", "a2"],
        gene_identifiers=list(vocab.genes[:3]),
        gene_symbols=["A", "B", "C"],
        vocabulary=vocab,
        sample_name="mock-sham-L",
    )
    b = SGMAECellDataset(
        expression=sparse.csr_matrix([[9., 0., 1., 3.]]),
        cell_ids=["b1"],
        gene_identifiers=[vocab.genes[i] for i in (3, 1, 0, 5)],
        gene_symbols=["D", "B", "A", "F"],
        vocabulary=vocab,
        sample_name="mock-sham-R",
    )
    loader = DataLoader(
        ConcatDataset([a, b]), batch_size=3, shuffle=False,
        collate_fn=collate_sgmae_batch, num_workers=0,
    )
    batch = next(iter(loader))
    reconstruction_mask = generate_reconstruction_mask(
        batch["measured_mask"], mask_ratio=0.5,
        gene_ids=batch["gene_ids"],
        generator=torch.Generator().manual_seed(17),
    )
    model = GeneAwareMAE(
        vocab_size=len(vocab), token_dim=128, cell_dim=128,
        num_latents=8, num_heads=4, dropout=0.0,
    )
    outputs = model(
        expression=batch["expression"],
        gene_ids=batch["gene_ids"],
        measured_mask=batch["measured_mask"],
        reconstruction_mask=reconstruction_mask,
    )
    assert torch.equal(outputs["target_reconstruction_mask"], reconstruction_mask)
    assert torch.equal(outputs["target_counts"], batch["expression"])
    assert not (reconstruction_mask & ~batch["measured_mask"]).any().item()
    assert torch.isfinite(outputs["cell_embeddings"]).all().item()
    print("Dataset -> masking -> SG-MAE integration: PASS")


def main():
    torch.manual_seed(123)
    test_basic_masking()
    test_dynamic_panels()
    test_measured_zero_and_no_leakage()
    test_repeatability()
    test_mask_fraction_edges()
    test_invalid_inputs()
    test_old_api()
    test_large_panel()
    test_model_and_dataset_integration()
    print("\nAll SG-MAE dynamic masking tests passed.")


if __name__ == "__main__":
    main()
