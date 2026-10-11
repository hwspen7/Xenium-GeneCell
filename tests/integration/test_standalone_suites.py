"""Execute standalone synthetic regression suites under pytest and CI.

These suites were previously available only via ``python -m scripts.test_*``.
Real-data inspection utilities ``test_data_loading`` and ``test_dataset`` are
not part of automated CI because they require local Xenium files.
"""

from importlib import import_module

import pytest


@pytest.mark.parametrize(
    "module_name, entrypoint",
    [
        ("test_cell_representations", "main_test"),
        ("test_correlation_plots", "main"),
        ("test_cross_condition", "main"),
        ("test_evaluation_entrypoint", "main"),
        ("test_gene_vocabulary", "main"),
        ("test_losses", "main"),
        ("test_scalable_model", "main"),
        ("test_scheduler", "main"),
        ("test_scheduler_entrypoint", "main"),
        ("test_sgmae_dataset", "main"),
        ("test_train_entrypoint", "main"),
        ("test_trainer", "main"),
    ],
)
def test_standalone_suite(module_name: str, entrypoint: str) -> None:
    """Keep former command-line smoke/regression checks in standard test discovery."""
    module = import_module(f"tests.v1.{module_name}")
    getattr(module, entrypoint)()
