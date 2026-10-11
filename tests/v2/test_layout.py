"""Check that the V2 experiment namespace is importable.

Spatial graph/GNN behavior will be tested after the V2 source is added.
"""

import importlib


def test_v2_experiment_namespace():
    module = importlib.import_module("scripts.v2")
    assert module.__name__ == "scripts.v2"
