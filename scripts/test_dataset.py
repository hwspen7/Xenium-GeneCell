
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import load_xenium_samples


def main():
    data_root = Path("data")

    datasets, gene_names = load_xenium_samples(data_root)

    print("Xenium Dataset Test")
    print("=" * 40)

    total_cells = 0

    for name, dataset in datasets.items():
        print(f"\nSample: {name}")
        print(f"Cells: {len(dataset):,}")
        print(f"Genes: {len(dataset.gene_names)}")
        print(f"Nonzero entries: {dataset.expression.nnz:,}")

        assert dataset.expression.shape[1] == 300
        assert dataset.gene_names == gene_names

        loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
        )

        batch = next(iter(loader))

        assert batch.shape == (8, 300)
        assert batch.dtype == torch.float32
        assert torch.isfinite(batch).all()
        assert (batch >= 0).all()

        print(f"Batch shape: {tuple(batch.shape)}")
        print("Status: PASSED")

        total_cells += len(dataset)

    print("\n" + "=" * 40)
    print(f"Samples: {len(datasets)}")
    print(f"Total Cells: {total_cells:,}")
    print(f"Aligned Genes: {len(gene_names)}")
    print("\nAll dataset tests passed.")


if __name__ == "__main__":
    main()
