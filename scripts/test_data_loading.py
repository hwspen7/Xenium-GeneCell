
from pathlib import Path

from src.dataset import load_xenium_expression


def main():
    data_root = Path("data")
    files = sorted(data_root.rglob("cell_feature_matrix.h5"))

    if not files:
        raise FileNotFoundError(
            "No cell_feature_matrix.h5 found in data/"
        )

    for file_path in files:
        print(f"\nLoading: {file_path.parent.name}")

        expression, cell_ids, gene_names = (
            load_xenium_expression(file_path)
        )

        print(f"Cells: {expression.shape[0]:,}")
        print(f"Genes: {expression.shape[1]}")
        print(f"Nonzero entries: {expression.nnz:,}")

        assert expression.shape[0] == len(cell_ids)
        assert expression.shape[1] == len(gene_names)

        print("Status: PASSED")

    print("\nAll datasets loaded successfully.")


if __name__ == "__main__":
    main()
