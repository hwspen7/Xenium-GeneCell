import pytest
import torch

from cellmae.device import select_device


def test_cpu_is_available():
    assert select_device("cpu") == torch.device("cpu")


def test_auto_selects_a_valid_backend():
    assert select_device("auto").type in {"cuda", "mps", "cpu"}


def test_invalid_device_rejected():
    with pytest.raises(ValueError):
        select_device("invalid")
