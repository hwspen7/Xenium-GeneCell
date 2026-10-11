
import hashlib
import json
import os
import tempfile

from numbers import Integral
from pathlib import Path
from types import MappingProxyType


class GeneVocabulary:
    """
    Stable and extensible gene vocabulary for SG-MAE.

    Each gene receives a persistent integer identifier.
    Existing identifiers are preserved during expansion.

    Gene ID -1 is reserved for padding.
    Gene identifiers are case-sensitive.
    """

    SCHEMA_VERSION = 2
    PAD_ID = -1

    def __init__(
        self,
        genes: list[str],
        species: str = "Mus musculus",
        namespace: str = "gene_symbol",
    ):
        self._validate_metadata(species, "species")
        self._validate_metadata(namespace, "namespace")

        validated = self._validate_genes(genes)

        if not validated:
            raise ValueError(
                "Vocabulary must contain at least one gene."
            )

        self.species = species
        self.namespace = namespace

        self._genes = tuple(validated)

        self._gene_to_id = {
            gene: index
            for index, gene in enumerate(self._genes)
        }

    @staticmethod
    def _validate_metadata(value: str, field: str):
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
        ):
            raise ValueError(f"Invalid {field}.")

    @staticmethod
    def _validate_genes(genes) -> list[str]:
        """Validate gene identifiers and reject duplicates."""

        if not isinstance(genes, (list, tuple)):
            raise TypeError(
                "Genes must be a list or tuple."
            )

        for gene in genes:
            if (
                not isinstance(gene, str)
                or not gene
                or gene != gene.strip()
            ):
                raise ValueError(
                    f"Invalid gene identifier: {gene!r}"
                )

        if len(genes) != len(set(genes)):
            raise ValueError(
                "Duplicate gene identifiers detected."
            )

        return list(genes)

    @property
    def genes(self) -> tuple[str, ...]:
        """Return gene identifiers in stable ID order."""
        return self._genes

    @property
    def gene_to_id(self):
        """Expose a read-only gene-to-ID mapping."""
        return MappingProxyType(self._gene_to_id)

    @classmethod
    def from_genes(
        cls,
        genes: list[str],
        species: str = "Mus musculus",
        namespace: str = "gene_symbol",
    ):
        """Create a deterministically ordered vocabulary."""

        validated = cls._validate_genes(genes)

        return cls(
            genes=sorted(validated),
            species=species,
            namespace=namespace,
        )

    def __len__(self):
        return len(self._genes)

    def encode(
        self,
        genes: list[str],
        allow_unknown: bool = False,
    ) -> list[int]:
        """
        Convert gene identifiers to integer IDs.

        Unknown genes raise an error by default.

        When allow_unknown is True, unknown genes map to -1.
        This option must not be used for measured targets.
        """

        validated = self._validate_genes(genes)

        result = []

        for gene in validated:
            if gene not in self._gene_to_id:
                if not allow_unknown:
                    raise KeyError(
                        f"Unknown gene identifier: {gene}"
                    )

                result.append(self.PAD_ID)
            else:
                result.append(self._gene_to_id[gene])

        return result

    def decode(self, ids) -> list[str | None]:
        """Convert integer IDs back to gene identifiers."""

        decoded = []

        for gene_id in ids:
            if (
                isinstance(gene_id, bool)
                or not isinstance(gene_id, Integral)
            ):
                raise TypeError(
                    "Gene IDs must be integers."
                )

            gene_id = int(gene_id)

            if gene_id == self.PAD_ID:
                decoded.append(None)

            elif 0 <= gene_id < len(self._genes):
                decoded.append(self._genes[gene_id])

            else:
                raise ValueError(
                    f"Gene ID outside vocabulary: {gene_id}"
                )

        return decoded

    def extend(self, new_genes: list[str]) -> list[str]:
        """
        Append new genes while preserving existing IDs.

        Newly discovered genes are appended in sorted order.
        """

        validated = self._validate_genes(new_genes)

        additions = sorted(
            gene
            for gene in validated
            if gene not in self._gene_to_id
        )

        if not additions:
            return []

        start = len(self._genes)

        self._genes = self._genes + tuple(additions)

        self._gene_to_id.update({
            gene: start + offset
            for offset, gene in enumerate(additions)
        })

        return additions

    def to_dict(self) -> dict:
        """Return canonical vocabulary metadata."""

        return {
            "schema_version": self.SCHEMA_VERSION,
            "species": self.species,
            "namespace": self.namespace,
            "genes": list(self._genes),
        }

    @staticmethod
    def _hash_payload(payload: dict) -> str:
        """Calculate a deterministic SHA-256 fingerprint."""

        canonical = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        return hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()

    @property
    def fingerprint(self) -> str:
        """Return the fingerprint of the current vocabulary."""

        return self._hash_payload(self.to_dict())

    def save(self, file_path: str | Path) -> None:
        """Atomically save a fingerprinted JSON vocabulary."""

        path = Path(file_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = self.to_dict()
        payload["fingerprint"] = self.fingerprint

        temporary_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)

                json.dump(
                    payload,
                    stream,
                    indent=2,
                    ensure_ascii=False,
                )

                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())

            os.replace(temporary_path, path)

        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, file_path: str | Path):
        """Load a vocabulary and verify its fingerprint."""

        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Vocabulary not found: {path}"
            )

        payload = json.loads(
            path.read_text(encoding="utf-8")
        )

        required = {
            "schema_version",
            "species",
            "namespace",
            "genes",
            "fingerprint",
        }

        if not isinstance(payload, dict):
            raise ValueError("Invalid vocabulary file.")

        if set(payload) != required:
            raise ValueError(
                "Invalid vocabulary file fields."
            )

        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != cls.SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported vocabulary schema version."
            )

        supplied_fingerprint = payload.pop(
            "fingerprint"
        )

        expected_fingerprint = cls._hash_payload(
            payload
        )

        if (
            not isinstance(supplied_fingerprint, str)
            or supplied_fingerprint != expected_fingerprint
        ):
            raise ValueError(
                "Vocabulary fingerprint mismatch."
            )

        return cls(
            genes=payload["genes"],
            species=payload["species"],
            namespace=payload["namespace"],
        )
