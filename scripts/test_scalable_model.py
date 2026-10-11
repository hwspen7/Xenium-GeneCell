"""Targeted architecture and leakage regression tests for SG-MAE V1.0."""

import torch
import torch.nn.functional as F
from src.models.gene_aware_mae import GeneAwareMAE


def expect_raises(error_type, fn):
    try:
        fn()
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__}")


def main():
    torch.manual_seed(42)
    m = GeneAwareMAE(vocab_size=6000, dropout=0.0)
    m.eval()
    x = torch.randint(0, 10, (4, 300)).float()
    ids = torch.arange(300)
    measured = torch.ones_like(x, dtype=torch.bool)
    mask = torch.rand_like(x) < 0.20

    result = m(x, ids, measured, mask)
    assert result['cell_embeddings'].shape == (4, 128)
    assert result['presence_logits'].shape == (4, 300)
    assert result['positive_intensity'].shape == (4, 300)
    assert torch.equal(result['target_reconstruction_mask'], mask)
    assert torch.equal(result['target_counts'], x)
    print('Architecture 300 genes: PASS')

    changed = x.clone()
    changed[mask] = changed[mask] + 123456
    out_changed = m(changed, ids, measured, mask)
    for key in ('cell_embeddings', 'presence_logits', 'positive_intensity'):
        torch.testing.assert_close(result[key], out_changed[key], atol=0, rtol=0)
    print('Masked-count leakage: PASS')

    # Reordering columns and gene IDs together must not change predictions.
    perm = torch.randperm(300)
    out_perm = m(x[:, perm], ids[perm], measured[:, perm], mask[:, perm], target_gene_ids=ids)
    for key in ('cell_embeddings', 'presence_logits', 'positive_intensity'):
        torch.testing.assert_close(result[key], out_perm[key], atol=2e-5, rtol=2e-5)
    print('Gene permutation invariance: PASS')

    targets = torch.tensor([255, 30, 900, -1, 5, 35], dtype=torch.long)
    subset = m(x, ids, measured, mask, target_gene_ids=targets)
    assert subset['presence_logits'].shape == (4, 6)
    for idx, gene_id in enumerate(targets.tolist()):
        if 0 <= gene_id < 300:
            assert torch.equal(subset['target_counts'][:,idx], x[:,gene_id])
            assert torch.equal(subset['target_reconstruction_mask'][:,idx], mask[:,gene_id])
            assert subset['target_measured_mask'][:,idx].all()
        else:
            assert not subset['target_reconstruction_mask'][:,idx].any()
            assert not subset['target_measured_mask'][:,idx].any()
            assert subset['target_counts'][:,idx].eq(0).all()
    torch.testing.assert_close(subset['presence_logits'][:, 0], result['presence_logits'][:, 255])
    print('Target alignment (reorder/unmeasured/pad): PASS')

    duplicates = ids.clone()
    duplicates[2] = duplicates[1]
    expect_raises(ValueError, lambda: m(x, duplicates, measured, mask))
    expect_raises(ValueError, lambda: m(x, ids, measured, mask, target_gene_ids=torch.tensor([3,3])))
    print('Duplicate valid Gene IDs rejected: PASS')

    x_bad = x.clone()
    x_bad[:,100] = float('nan')
    absent = measured.clone()
    absent[:,100] = False
    assert torch.isfinite(m(x_bad, ids, absent, mask & absent)['cell_embeddings']).all()
    print('Unmeasured and NaN values excluded: PASS')

    all_hidden = measured.clone()
    z = m(x, ids, measured, all_hidden)
    assert torch.isfinite(z['cell_embeddings']).all()
    torch.testing.assert_close(z['cell_embeddings'][0], z['cell_embeddings'][1])
    print('All-hidden fallback: PASS')

    variable_ids = torch.tensor([[8,4,-1,-1,10],[11,12,9,-1,-1]])
    xm = torch.tensor([[1.,0.,float('nan'),float('nan'),4.],[2.,3.,0.,float('nan'),float('nan')]])
    mm = torch.tensor([[True, True, False, False, True],[True,True,True,False,False]])
    rm = torch.tensor([[False,True,False,False,True],[True,False,False,False,False]])
    vr = m(xm,variable_ids,mm,rm,target_gene_ids=torch.tensor([[10,8,90,-1],[9,12,8,-1]]))
    assert vr['target_reconstruction_mask'].tolist() == [[True,False,False,False],[False,False,False,False]]
    assert torch.isfinite(vr['cell_embeddings']).all()
    print('Variable gene sets and padding: PASS')

    m.train()
    output = m(x, ids, measured, mask, target_gene_ids=torch.tensor([0,1,2,3,4,5,6,7,8,9]))
    active = output['target_reconstruction_mask']
    target = output['target_counts']
    loss = F.binary_cross_entropy_with_logits(output['presence_logits'][active], (target[active] > 0).float())
    positive = active & (target > 0)
    loss = loss + F.smooth_l1_loss(output['positive_intensity'][positive], torch.log1p(target[positive]))
    loss.backward()
    for name in ('gene_embeddings.weight','latent_queries','cross_attention.in_proj_weight','latent_block.attention.in_proj_weight','cell_projection.weight','presence_projection.weight','intensity_projection.weight'):
        g = dict(m.named_parameters())[name].grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, name
    print('BCE/Huber training backward gradients: PASS')

    x_large = torch.randint(0, 5, (2, 5000)).float()
    id_large = torch.arange(5000)
    measured_large = torch.ones_like(x_large, dtype=torch.bool)
    masked_large = torch.rand_like(x_large) < .2
    m.eval()
    with torch.no_grad():
        large = m(x_large, id_large, measured_large, masked_large, target_gene_ids=id_large[:100])
    assert large['cell_embeddings'].shape == (2,128)
    assert large['presence_logits'].shape == (2,100)
    print('Architecture 5000 genes, target sampling: PASS')
    expand_model = GeneAwareMAE(vocab_size=300, dropout=0.0).eval()
    small_counts = x[:2, :30]
    small_ids = torch.arange(30)
    small_measured = torch.ones_like(small_counts, dtype=torch.bool)
    small_mask = torch.zeros_like(small_measured)
    before = expand_model(small_counts, small_ids, small_measured, small_mask)
    expand_model.expand_vocabulary(6000)
    after = expand_model(small_counts, small_ids, small_measured, small_mask)
    for key in ('cell_embeddings', 'presence_logits', 'positive_intensity'):
        torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
    print('Vocabulary expansion preserves previous predictions: PASS')

    print('All SG-MAE core regression tests passed.')

if __name__ == '__main__':
    main()
