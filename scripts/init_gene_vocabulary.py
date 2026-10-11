"""Explicit, non-destructive initialization of ShamL's stable Gene Vocabulary."""

import argparse
from pathlib import Path

from src.dataset import find_xenium_sample_file, initialize_sham_vocabulary


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root / "data")
    parser.add_argument("--train-sample", default="ShamL")
    parser.add_argument(
        "--output", type=Path, default=root / "configs" / "gene_vocabulary.json"
    )
    args = parser.parse_args()

    source = find_xenium_sample_file(args.data_root, args.train_sample)
    existed = args.output.exists()
    vocabulary = initialize_sham_vocabulary(source, args.output)
    print("Gene Vocabulary:", "EXISTING VERIFIED" if existed else "CREATED")
    print("Source:", source)
    print("Output:", args.output)
    print("Genes:", len(vocabulary))
    print("Namespace:", vocabulary.namespace)
    print("Fingerprint:", vocabulary.fingerprint)


if __name__ == "__main__":
    main()
