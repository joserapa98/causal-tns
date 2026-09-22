"""Tests for compiled causal query views."""

import itertools

import pytest
import torch
import torch.nn.functional as nnf

from models import CausalChain, CausalDAG


def test_plan_shares_parameters_and_logical_edges() -> None:
    chain = CausalChain(2, 2, 3, directions="r")
    params_before = tuple(id(parameter) for parameter in chain.parameters())
    plan = chain.compile_query([0])
    parent, child = chain.causal_nodes.values()
    parent_view, child_view = plan.views
    edge = chain.logical_edges[0]

    assert parent_view.k_node.tensor is parent.k_node.tensor
    assert parent_view.out_nodes["x1"].tensor is parent.out_nodes["x1"].tensor
    assert child_view.k_node.tensor is child.k_node.tensor
    assert parent_view.out_nodes["x1"]["bond"] is edge
    assert child_view.k_node["in_x0"] is edge
    assert chain.compile_query([0]) is plan
    assert tuple(id(parameter) for parameter in chain.parameters()) == params_before
    assert len(parent_view.input_edges) == 1


def test_ancestor_pruning_and_partial_views() -> None:
    dag = CausalDAG(
        [[0, 2, 2, 0], [0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0]],
        2,
        dtype=torch.float64,
    )
    plan = dag.compile_query([0, 1, 2])

    # x0 and x1 stay active because x3 is observed; x2 is pruned.
    assert plan.active_features == (0, 1, 3)
    assert len(plan.active_edges) == 2
    assert tuple(plan.views[0].out_nodes) == ("x1",)
    assert tuple(plan.views[1].out_nodes) == ("x3",)
    assert plan.views[0].copy_node is not None
    assert len(plan.views[0].copy_node.axes_names) == 3

    full_plan = dag.compile_query()
    states = torch.tensor(list(itertools.product(range(2), repeat=4)))
    full_input = nnf.one_hot(states, num_classes=2).to(torch.float64)
    full = dag.evaluate(full_input, plan=full_plan)
    input = torch.eye(2, dtype=torch.float64)
    marginal = dag.evaluate([input], plan=plan)
    assert torch.allclose(
        marginal,
        torch.stack([full[states[:, 3] == value].sum() for value in range(2)]),
    )


def test_plans_reuse_views_and_gradients() -> None:
    chain = CausalChain(3, 2, 2, directions="rl", dtype=torch.float64)
    states = torch.tensor(list(itertools.product(range(2), repeat=3)))
    input = nnf.one_hot(states, num_classes=2).to(torch.float64)
    pairs = torch.tensor(list(itertools.product(range(2), repeat=2)))
    pair_input = nnf.one_hot(pairs, num_classes=2).to(torch.float64)
    plan = chain.compile_query([0])
    other = chain.compile_query([2])
    full = chain.compile_query()

    for _ in range(3):
        a = chain.evaluate(pair_input, plan=plan)
        b = chain.evaluate(pair_input, plan=other)
        c = chain.evaluate(input, plan=full)
        assert torch.allclose(a.sum(), torch.ones((), dtype=torch.float64))
        assert torch.allclose(b.sum(), torch.ones((), dtype=torch.float64))
        assert torch.allclose(c.sum(), torch.ones((), dtype=torch.float64))

    (a[0] + b[0] + c[0]).backward()
    assert all(parameter.grad is not None for parameter in chain.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in chain.parameters())
    assert any(torch.any(parameter.grad != 0) for parameter in chain.parameters())
    assert chain.normalize().item() == 1


def test_local_stochastic_factors_match_explicit_distribution() -> None:
    chain = CausalChain(2, 2, 2, directions="r", dtype=torch.float64)
    parent, child = chain.causal_nodes.values()
    k_parent = torch.tensor([0.3, 0.7], dtype=torch.float64)
    b = torch.tensor([[0.8, 0.2], [0.1, 0.9]], dtype=torch.float64)
    k_child = torch.tensor([[0.9, 0.2], [0.1, 0.8]], dtype=torch.float64)
    parent.k_node.set_tensor(k_parent.sqrt())
    parent.out_nodes["x1"].set_tensor(b.sqrt())
    child.k_node.set_tensor(k_child.sqrt())

    states = torch.tensor(list(itertools.product(range(2), repeat=2)))
    input = nnf.one_hot(states, num_classes=2).to(torch.float64)
    expected = torch.stack(
        [k_parent[x] * (b[x] * k_child[y]).sum() for x, y in states.tolist()]
    )
    assert torch.allclose(chain.evaluate(input), expected)
    assert torch.allclose(expected.sum(), torch.ones((), dtype=torch.float64))


def test_zero_parameter_rows_stay_normalized() -> None:
    chain = CausalChain(2, 2, 2, directions="r", init_method="zeros")
    values = torch.tensor(list(itertools.product(range(2), repeat=2)))
    input = nnf.one_hot(values, num_classes=2).to(torch.get_default_dtype())

    probabilities = chain.evaluate(input)
    assert torch.all(torch.isfinite(probabilities))
    assert torch.allclose(probabilities, torch.full((4,), 0.25))


def test_plan_survives_dtype_conversion() -> None:
    chain = CausalChain(2, 2, 2)
    plan = chain.compile_query([0])
    chain.to(torch.float64)

    probabilities = chain.evaluate([torch.eye(2, dtype=torch.float64)], plan=plan)
    assert probabilities.dtype == torch.float64
    assert torch.allclose(probabilities.sum(), torch.ones((), dtype=torch.float64))

    other = CausalChain(2, 2, 2)
    other.to(torch.float64)
    other_plan = other.compile_query([0])
    result = other.evaluate([torch.eye(2, dtype=torch.float64)], plan=other_plan)
    assert result.dtype == torch.float64


def test_query_plan_validation() -> None:
    chain = CausalChain(2, 2, 2)
    other = CausalChain(2, 2, 2)
    plan = chain.compile_query([1])
    input = torch.eye(2)

    with pytest.raises(ValueError, match="either"):
        chain.evaluate([input], marg_features=[1], plan=plan)
    with pytest.raises(ValueError, match="belong"):
        other.evaluate([input], plan=plan)
