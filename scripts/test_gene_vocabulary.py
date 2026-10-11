import json
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from src.data.gene_vocabulary import GeneVocabulary


def main():
    genes = [
        "Spp1",
        "Aqp2",
        "Havcr1",
        "Aqp1",
    ]

    vocabulary = GeneVocabulary.from_genes(genes)

    assert len(vocabulary) == 4

    assert vocabulary.encode([
        "Aqp1",
        "Aqp2",
        "Havcr1",
        "Spp1",
    ]) == [0, 1, 2, 3]

    print("Initial gene mapping: PASS")

    shuffled = [
        "Spp1",
        "Aqp1",
        "Havcr1",
        "Aqp2",
    ]

    assert vocabulary.encode(shuffled) == [3, 0, 2, 1]

    print("Gene ordering independence: PASS")

    previous_ids = {
        gene: vocabulary.encode([gene])[0]
        for gene in genes
    }

    additions = vocabulary.extend([
        "Umod",
        "Col1a1",
    ])

    assert additions == ["Col1a1", "Umod"]
    assert len(vocabulary) == 6

    for gene, gene_id in previous_ids.items():
        assert vocabulary.encode([gene])[0] == gene_id

    assert vocabulary.encode(["Col1a1", "Umod"]) == [4, 5]

    print("Vocabulary expansion: PASS")

    assert vocabulary.decode([0, 4, -1]) == [
        "Aqp1",
        "Col1a1",
        None,
    ]

    print("Gene ID decoding: PASS")

    assert vocabulary.encode(
        ["UnknownGene"],
        allow_unknown=True,
    ) == [-1]

    try:
        vocabulary.encode(["UnknownGene"])
    except KeyError:
        pass
    else:
        raise AssertionError("Unknown gene was accepted.")

    print("Unknown gene handling: PASS")

    try:
        GeneVocabulary.from_genes(["Aqp1", "Aqp1"])
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate gene was accepted.")

    print("Duplicate gene validation: PASS")

    with TemporaryDirectory() as directory:
        path = Path(directory) / "gene_vocabulary.json"

        vocabulary.save(path)

        restored = GeneVocabulary.load(path)

        assert restored.to_dict() == vocabulary.to_dict()
        assert restored.fingerprint == vocabulary.fingerprint

    print("Vocabulary serialization: PASS")
    print("Vocabulary fingerprint: PASS")


    # Verify that public gene ordering cannot be modified.
    try:
        vocabulary.genes.reverse()
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "Gene ordering must be read-only."
        )

    # Verify that the ID mapping cannot be modified.
    try:
        vocabulary.gene_to_id["Aqp1"] = 999
    except TypeError:
        pass
    else:
        raise AssertionError(
            "Gene ID mapping must be read-only."
        )

    print("Vocabulary immutability: PASS")

    # Verify repeated expansion does not change IDs.
    fingerprint_before = vocabulary.fingerprint

    additions = vocabulary.extend(["Col1a1", "Umod"])

    assert additions == []
    assert vocabulary.fingerprint == fingerprint_before

    print("Idempotent expansion: PASS")

    # Verify NumPy integer compatibility.
    assert vocabulary.decode(
        [np.int64(0)]
    ) == ["Aqp1"]

    print("NumPy integer decoding: PASS")

    # Verify rejection of empty vocabularies.
    try:
        GeneVocabulary.from_genes([])
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Empty vocabulary was accepted."
        )

    print("Empty vocabulary validation: PASS")

    # Verify that corrupted vocabulary files are rejected.
    with TemporaryDirectory() as directory:
        path = Path(directory) / "vocabulary.json"

        vocabulary.save(path)

        payload = json.loads(
            path.read_text(encoding="utf-8")
        )

        payload["genes"][0] = "ChangedGene"

        path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

        try:
            GeneVocabulary.load(path)
        except ValueError as error:
            assert "fingerprint" in str(error)
        else:
            raise AssertionError(
                "Modified vocabulary was accepted."
            )

    print("Vocabulary integrity validation: PASS")


    print("\nAll Gene Vocabulary tests passed.")




if __name__ == "__main__":
    main()
