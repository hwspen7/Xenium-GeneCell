from cellmae.cli import main, smoke_test
import torch


def test_doctor_cpu(capsys):
    assert main(["doctor", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "Selected device: cpu" in out


def test_synthetic_training_step_cpu():
    loss = smoke_test(torch.device("cpu"))
    assert 0.0 < loss < 10.0
