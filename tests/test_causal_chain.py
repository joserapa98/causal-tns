"""Tests for one-dimensional causal tensor networks."""

import itertools

import pytest
import torch
import torch.nn.functional as nnf

from models import CausalChain


@pytest.mark.parametrize(
    "directions",
    list(itertools.product(("->", "<-"), repeat=2)),
)
def test_probabilities_and_partial_marginals(
    directions: tuple[str, str],
) -> None:
    chain = CausalChain(
        n_features=3,
        phys_dim=2,
        bond_dim=3,
        directions=directions,
        dtype=torch.float64,
    )
    states = torch.tensor(list(itertools.product(range(2), repeat=3)))
    input = nnf.one_hot(states, num_classes=2).to(torch.float64)
    probabilities = chain.evaluate(input)

    assert torch.allclose(probabilities.sum(), torch.ones((), dtype=torch.float64))
    assert torch.all(probabilities >= 0)

    for n_marg in range(1, 4):
        for marg_features in itertools.combinations(range(3), n_marg):
            in_features = [i for i in range(3) if i not in marg_features]
            if not in_features:
                assert torch.allclose(
                    chain.evaluate(None, marg_features=marg_features),
                    torch.ones((), dtype=torch.float64),
                )
                continue

            in_values = torch.tensor(
                list(itertools.product(range(2), repeat=len(in_features)))
            )
            marg_input = nnf.one_hot(in_values, num_classes=2).to(torch.float64)
            marginals = chain.evaluate(marg_input, marg_features=marg_features)
            for row, values in enumerate(in_values):
                mask = (states[:, in_features] == values).all(dim=1)
                assert torch.allclose(marginals[row], probabilities[mask].sum())


def test_heterogeneous_dimensions_and_sampling() -> None:
    chain = CausalChain(
        n_features=3,
        phys_dim=[2, 3, 2],
        bond_dim=[2, 4],
        directions=["<-", "->"],
    )
    generator = torch.Generator().manual_seed(0)
    samples = chain.sample(20, generator=generator)

    assert samples.shape == (20, 3)
    assert samples.dtype == torch.long
    assert torch.all((samples[:, 0] >= 0) & (samples[:, 0] < 2))
    assert torch.all((samples[:, 1] >= 0) & (samples[:, 1] < 3))
    assert torch.all((samples[:, 2] >= 0) & (samples[:, 2] < 2))


def test_sampling_matches_dense_distribution() -> None:
    torch.manual_seed(0)
    chain = CausalChain(3, 2, 2, directions=["->", "<-"])
    states = torch.tensor(list(itertools.product(range(2), repeat=3)))
    input = nnf.one_hot(states, num_classes=2).to(torch.get_default_dtype())
    expected = chain.evaluate(input)

    samples = chain.sample(5_000, generator=torch.Generator().manual_seed(1))
    flat_samples = samples[:, 0] * 4 + samples[:, 1] * 2 + samples[:, 2]
    frequencies = torch.bincount(flat_samples, minlength=8) / samples.shape[0]

    assert torch.allclose(frequencies, expected, atol=0.04, rtol=0)


def test_single_variable_chain() -> None:
    chain = CausalChain(1, 3, 2, dtype=torch.float64)
    input = torch.eye(3, dtype=torch.float64).unsqueeze(1)

    assert torch.allclose(
        chain.evaluate(input).sum(), torch.ones((), dtype=torch.float64)
    )
    assert torch.allclose(
        chain.evaluate(None, marg_features=[0]),
        torch.ones((), dtype=torch.float64),
    )


def test_multiple_batch_axes() -> None:
    chain = CausalChain(2, 2, 2, n_batches=2)
    values = torch.randint(2, (3, 4, 2))
    input = nnf.one_hot(values, num_classes=2).to(torch.get_default_dtype())

    assert chain.evaluate(input).shape == (3, 4)
    with pytest.raises(RuntimeError, match="n_batches=1"):
        chain.sample(2)


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="n_features"):
        CausalChain(0, 2, 2)
    with pytest.raises(ValueError, match="directions"):
        CausalChain(3, 2, 2, directions=["->"])

    chain = CausalChain(3, 2, 2)
    with pytest.raises(ValueError, match="duplicates"):
        chain.evaluate(None, marg_features=[0, 0])
    with pytest.raises(ValueError, match="required"):
        chain.evaluate(None, marg_features=[0])
    with pytest.raises(ValueError, match="input features"):
        chain.evaluate(torch.zeros(4, 1, 2))


@pytest.mark.parametrize("directions", ["rlr", ["rlr"], ["r", "l", "r"]])
def test_compact_directions(directions: str | list[str]) -> None:
    chain = CausalChain(4, 2, 2, directions=directions)

    assert chain.directions == ("->", "<-", "->")
