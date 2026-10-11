"""Regression tests for SG-MAE V1.0 masked dual-head reconstruction loss."""

import torch
from torch.nn import functional as F

from src.training.losses import MaskedReconstructionLoss
from src.models.gene_aware_mae import GeneAwareMAE
from src.data.masking import generate_reconstruction_mask


def expect_raises(error_type, fn):
    try:
        fn()
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__}.")


def make_outputs():
    return {
        "presence_logits": torch.tensor(
            [[0.4, -0.2, 1.1, 123.0, float("nan"), -8.0]],
            requires_grad=True,
        ),
        "positive_intensity": torch.tensor(
            [[1.5, 0.4, 2.0, 100.0, float("nan"), -9.0]],
            requires_grad=True,
        ),
        "target_counts": torch.tensor(
            [[3.0, 0.0, 5.0, 9.0, float("nan"), 0.0]]
        ),
        "target_measured_mask": torch.tensor(
            [[True, True, True, True, False, False]]
        ),
        "target_reconstruction_mask": torch.tensor(
            [[True, True, True, False, False, False]]
        ),
        "target_valid_mask": torch.tensor(
            [[True, True, True, True, True, False]]
        ),
    }


def main():
    torch.manual_seed(42)
    objective = MaskedReconstructionLoss()
    outputs = make_outputs()
    result = objective(outputs)

    active = outputs["target_reconstruction_mask"]
    true_counts = outputs["target_counts"][active]
    positive = active & (outputs["target_counts"] > 0)
    expected_presence = F.binary_cross_entropy_with_logits(
        outputs["presence_logits"][active], (true_counts > 0).float()
    )
    expected_intensity = F.smooth_l1_loss(
        outputs["positive_intensity"][positive],
        torch.log1p(outputs["target_counts"][positive]),
        beta=1.0,
    )
    torch.testing.assert_close(result["presence_loss"], expected_presence)
    torch.testing.assert_close(result["intensity_loss"], expected_intensity)
    torch.testing.assert_close(
        result["total_loss"], expected_presence + expected_intensity
    )
    assert result["num_masked"].item() == 3
    assert result["num_positive"].item() == 2
    print("Masked BCE and positive-count Huber: PASS")

    # Measured zero must enter presence BCE, but not intensity Huber.
    zero_outputs = make_outputs()
    zero_outputs["target_reconstruction_mask"][:] = False
    zero_outputs["target_reconstruction_mask"][0, 1] = True
    zero_result = objective(zero_outputs)
    assert zero_result["num_masked"].item() == 1
    assert zero_result["num_positive"].item() == 0
    assert zero_result["intensity_loss"].item() == 0.0
    assert zero_result["presence_loss"].item() > 0
    print("Measured-zero targets and no positive targets: PASS")

    # No reconstruction targets must be finite and differentiable.
    empty_outputs = make_outputs()
    empty_outputs["target_reconstruction_mask"][:] = False
    empty_result = objective(empty_outputs)
    assert empty_result["total_loss"].item() == 0
    assert empty_result["total_loss"].requires_grad
    empty_result["total_loss"].backward()
    print("No masked targets differentiable zero: PASS")

    # Non-target predictions/counts may be changed without affecting the loss.
    altered = make_outputs()
    altered["target_counts"][0, 3] = 100000.0
    altered["presence_logits"] = altered["presence_logits"].clone()
    altered["positive_intensity"] = altered["positive_intensity"].clone()
    with torch.no_grad():
        altered["presence_logits"][0, 3] = -10000
        altered["positive_intensity"][0, 3] = 10000
    torch.testing.assert_close(objective(altered)["total_loss"], result["total_loss"])
    print("Unmasked, unmeasured, and padded targets excluded: PASS")

    # Gradients must reach both heads, but never non-target positions.
    result["total_loss"].backward()
    logits_grad = outputs["presence_logits"].grad
    intensity_grad = outputs["positive_intensity"].grad
    assert logits_grad[0, :3].abs().sum() > 0
    assert logits_grad[0, 3:].abs().sum() == 0
    assert intensity_grad[0, (0, 2)].abs().sum() > 0
    assert intensity_grad[0, 1].item() == 0
    assert intensity_grad[0, 3:].abs().sum() == 0
    print("Supervised-only gradients for both heads: PASS")

    weighted = MaskedReconstructionLoss(
        lambda_presence=2.0, lambda_intensity=0.25, huber_beta=0.5
    )
    weighted_result = weighted(make_outputs())
    torch.testing.assert_close(
        weighted_result["total_loss"],
        2.0 * weighted_result["presence_loss"]
        + 0.25 * weighted_result["intensity_loss"],
    )
    print("Configurable weights and Huber beta: PASS")

    invalid = make_outputs()
    invalid["target_reconstruction_mask"][0, 4] = True
    expect_raises(ValueError, lambda: objective(invalid))
    invalid = make_outputs()
    invalid["target_measured_mask"][0, 5] = True
    expect_raises(ValueError, lambda: objective(invalid))
    invalid = make_outputs()
    invalid["target_counts"][0, 1] = -1
    expect_raises(ValueError, lambda: objective(invalid))
    invalid = make_outputs()
    invalid["target_counts"][0, 1] = float("nan")
    expect_raises(ValueError, lambda: objective(invalid))
    invalid = make_outputs()
    with torch.no_grad():
        invalid["presence_logits"][0, 0] = float("nan")
    expect_raises(ValueError, lambda: objective(invalid))
    invalid = make_outputs()
    invalid["target_valid_mask"] = invalid["target_valid_mask"].float()
    expect_raises(TypeError, lambda: objective(invalid))
    expect_raises(ValueError, lambda: MaskedReconstructionLoss(-1.0, 1.0))
    expect_raises(ValueError, lambda: MaskedReconstructionLoss(0.0, 0.0))
    expect_raises(ValueError, lambda: MaskedReconstructionLoss(huber_beta=0))
    print("Invalid targets, masks, and hyperparameters rejected: PASS")

    # The model aligns reordered target gene IDs before the loss is calculated.
    model = GeneAwareMAE(vocab_size=64, dropout=0.0)
    x = torch.tensor([
        [2., 0., 6., 4., 0., 1.],
        [0., 3., 0., 1., 9., 2.],
        [5., 0., 1., 0., 1., 0.],
    ])
    ids = torch.tensor([31, 8, 40, 3, 22, 4])
    measured = torch.ones_like(x, dtype=torch.bool)
    reconstruction_mask = torch.tensor([
        [True, True, False, True, False, True],
        [True, False, True, False, True, False],
        [False, True, True, False, False, True],
    ])
    target_a = torch.tensor([4, 31, -1, 8, 40, 3, 22])
    pred_a = model(x, ids, measured, reconstruction_mask, target_gene_ids=target_a)
    loss_a = objective(pred_a)
    target_b = torch.tensor([22, 8, 3, 4, 40, 31, -1])
    pred_b = model(x, ids, measured, reconstruction_mask, target_gene_ids=target_b)
    loss_b = objective(pred_b)
    torch.testing.assert_close(loss_a["total_loss"], loss_b["total_loss"], atol=1e-6, rtol=1e-6)
    assert loss_a["num_masked"].item() == loss_b["num_masked"].item()
    print("Gene ID target reordering preserves loss: PASS")

    # Selected targets that were not measured cannot contribute to the objective.
    target_absent = torch.tensor([2, -1, 31, 4])
    pred_absent = model(x, ids, measured, reconstruction_mask, target_gene_ids=target_absent)
    assert not pred_absent["target_measured_mask"][:, :2].any()
    assert not pred_absent["target_reconstruction_mask"][:, :2].any()
    assert objective(pred_absent)["num_masked"].item() > 0
    print("Requested unmeasured and padding gene IDs ignored: PASS")

    # Full Dataset -> Masking -> Model -> Loss gradient flow.
    gene_ids = torch.arange(300)
    counts = torch.randint(0, 5, (5, 300)).float()
    # Inject guaranteed measured zeros without changing measurement status.
    counts[:, ::5] = 0
    measured = torch.ones_like(counts, dtype=torch.bool)
    measured[0, 290:] = False
    counts[0, 290:] = float("nan")
    recon = generate_reconstruction_mask(
        measured, mask_ratio=0.2, gene_ids=gene_ids,
        generator=torch.Generator(device="cpu").manual_seed(101)
    )
    large_model = GeneAwareMAE(vocab_size=300, dropout=0.0).train()
    output = large_model(counts, gene_ids, measured, recon)
    full_loss = objective(output)
    assert full_loss["num_masked"].item() == recon.sum().item()
    assert torch.isfinite(full_loss["total_loss"])
    full_loss["total_loss"].backward()
    for name in (
        "gene_embeddings.weight",
        "latent_queries",
        "cross_attention.in_proj_weight",
        "latent_block.attention.in_proj_weight",
        "cell_projection.weight",
        "presence_projection.weight",
        "intensity_projection.weight",
    ):
        grad = dict(large_model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all()
        assert grad.abs().sum() > 0, name
    print("Dataset-compatible masking -> SG-MAE -> Loss backward: PASS")

    print("\nAll SG-MAE dual-head loss tests passed.")


if __name__ == "__main__":
    main()
