import torch
import pytest
from train import PrunableLinear, SelfPruningNet


def test_output_shape():
    layer = PrunableLinear(10, 5)
    x = torch.randn(2, 10)
    assert layer(x).shape == (2, 5)


def test_grad_flow_weights_and_gates():
    layer = PrunableLinear(10, 5)
    x = torch.randn(2, 10)
    out = layer(x).sum()
    out.backward()
    assert layer.weight.grad is not None
    assert layer.gate_scores.grad is not None


def test_gate_range():
    layer = PrunableLinear(10, 5)
    gates = torch.sigmoid(layer.gate_scores)
    assert torch.all(gates > 0) and torch.all(gates < 1)


def test_zero_gate_equals_bias():
    layer = PrunableLinear(10, 5)
    layer.gate_scores.data.fill_(-100)
    x = torch.randn(2, 10)
    out = layer(x)
    assert torch.allclose(out, layer.bias.expand_as(out), atol=1e-4)


def test_full_gate_equals_linear():
    layer = PrunableLinear(10, 5)
    layer.gate_scores.data.fill_(100)
    x = torch.randn(2, 10)
    out1 = layer(x)
    out2 = torch.nn.functional.linear(x, layer.weight, layer.bias)
    assert torch.allclose(out1, out2, atol=1e-4)


def test_manual_forward():
    layer = PrunableLinear(10, 5)
    x = torch.randn(2, 10)
    gates = torch.sigmoid(layer.gate_scores)
    expected = torch.nn.functional.linear(x, layer.weight * gates, layer.bias)
    assert torch.allclose(layer(x), expected)


def test_model_output_shape():
    model = SelfPruningNet()
    x = torch.randn(4, 3, 32, 32)
    assert model(x).shape == (4, 10)


def test_sparsity_loss_positive():
    model = SelfPruningNet()
    loss = model.sparsity_loss()
    assert loss.item() > 0


def test_sparsity_loss_grad():
    model = SelfPruningNet()
    loss = model.sparsity_loss()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_global_sparsity_bounds():
    model = SelfPruningNet()
    s = model.global_sparsity(1e-2)
    assert 0 <= s <= 100


def test_lambda_warmup_monotonic():
    vals = [min(1.0, i / 5) for i in range(10)]
    assert all(vals[i] <= vals[i+1] for i in range(len(vals)-1))


def test_param_groups():
    model = SelfPruningNet()
    gate_params = [p for n, p in model.named_parameters() if "gate_scores" in n]
    weight_params = [p for n, p in model.named_parameters() if "gate_scores" not in n]
    assert len(gate_params) > 0
    assert len(weight_params) > 0


def test_gate_extremes():
    layer = PrunableLinear(5, 3)
    layer.gate_scores.data.fill_(0)
    gates = torch.sigmoid(layer.gate_scores)
    assert torch.allclose(gates, torch.full_like(gates, 0.5))


def test_backward_consistency():
    layer = PrunableLinear(5, 3)
    x = torch.randn(2, 5, requires_grad=True)
    out = layer(x).sum()
    out.backward()
    assert x.grad is not None


def test_multiple_layers_sparsity():
    model = SelfPruningNet()
    s = model.global_sparsity(1e-2)
    assert isinstance(s, float)