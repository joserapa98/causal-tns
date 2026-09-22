"""Tests for causal tensor networks on directed acyclic graphs."""

import itertools

import pytest
import torch
import torch.nn.functional as nnf

from models import CausalChain, CausalDAG


def _diamond_dag() -> CausalDAG:
    adjacency = [
        [0, 2, 2, 0],
        [0, 0, 0, 2],
        [0, 0, 0, 2],
        [0, 0, 0, 0],
    ]
    return CausalDAG(adjacency, phys_dim=2, dtype=torch.float64)


def test_dag_probabilities_and_partial_marginals() -> None:
    dag = _diamond_dag()
    states = torch.tensor(list(itertools.product(range(2), repeat=4)))
    input = nnf.one_hot(states, num_classes=2).to(torch.float64)
    probabilities = dag.evaluate(input)

    assert torch.all(probabilities >= 0)
    assert torch.allclose(probabilities.sum(), torch.ones((), dtype=torch.float64))

    for marg_features in [(0,), (1, 2), (0, 3), (0, 1, 2, 3)]:
        in_features = [i for i in range(4) if i not in marg_features]
        if not in_features:
            assert torch.allclose(
                dag.evaluate(None, marg_features=marg_features),
                torch.ones((), dtype=torch.float64),
            )
            continue
        values = torch.tensor(
            list(itertools.product(range(2), repeat=len(in_features)))
        )
        marg_input = nnf.one_hot(values, num_classes=2).to(torch.float64)
        marginals = dag.evaluate(marg_input, marg_features=marg_features)
        for row, assignment in enumerate(values):
            mask = (states[:, in_features] == assignment).all(dim=1)
            assert torch.allclose(marginals[row], probabilities[mask].sum())


def test_dag_einsum_differentiates_and_samples() -> None:
    torch.manual_seed(0)
    dag = _diamond_dag()
    values = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
    input = nnf.one_hot(values, num_classes=2).to(torch.float64)

    loss = (
        dag.evaluate(input).sum()
        + dag.evaluate([input[:, 0], input[:, 3]], marg_features=[1, 2]).sum()
    )
    loss.backward()
    samples = dag.sample(20, generator=torch.Generator().manual_seed(0))

    assert all(parameter.grad is not None for parameter in dag.parameters())
    assert samples.shape == (20, 4)
    assert torch.all((samples >= 0) & (samples < 2))


def test_dag_weighted_adjacency() -> None:
    adjacency = [[0, 2, 3], [0, 0, 0], [0, 0, 0]]
    dag = CausalDAG(adjacency, phys_dim=[2, 3, 2])

    assert torch.equal(dag.adjacency, torch.tensor(adjacency))
    assert dag.logical_edges[0].size() == 2
    assert dag.logical_edges[1].size() == 3
    assert dag.topological_order == (0, 1, 2)


def test_dag_einsum_matches_chain_contraction() -> None:
    adjacency = [[0, 2, 0], [0, 0, 0], [0, 2, 0]]
    torch.manual_seed(2)
    chain = CausalChain(3, 2, 2, directions="rl", dtype=torch.float64)
    torch.manual_seed(2)
    dag = CausalDAG(adjacency, 2, dtype=torch.float64)
    states = torch.tensor(list(itertools.product(range(2), repeat=3)))
    input = nnf.one_hot(states, num_classes=2).to(torch.float64)

    assert all(
        torch.equal(tensor, dag.state_dict()[name])
        for name, tensor in chain.state_dict().items()
    )
    assert torch.allclose(chain.evaluate(input), dag.evaluate(input))
    assert torch.allclose(
        chain.evaluate([input[:, 1]], marg_features=[0, 2]),
        dag.evaluate([input[:, 1]], marg_features=[0, 2]),
    )


def test_dag_multiple_batch_axes() -> None:
    dag = CausalDAG([[0, 2], [0, 0]], 2, n_batches=2)
    values = torch.randint(2, (3, 4, 2))
    input = nnf.one_hot(values, num_classes=2).to(torch.get_default_dtype())

    assert dag.evaluate(input).shape == (3, 4)


@pytest.mark.parametrize(
    ("adjacency", "message"),
    [
        ([[0, 2], [3, 0]], "acyclic"),
        ([[1, 0], [0, 0]], "self-edges"),
        ([[0, -2], [0, 0]], "negative"),
        ([[0, 1, 0], [0, 0, 0]], "square"),
    ],
)
def test_invalid_adjacency(adjacency: list[list[int]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CausalDAG(adjacency, phys_dim=2)


def test_adjacency_requires_integer_bond_dimensions() -> None:
    with pytest.raises(TypeError, match="integer bond dimensions"):
        CausalDAG([[0, 2.0], [0, 0]], phys_dim=2)
