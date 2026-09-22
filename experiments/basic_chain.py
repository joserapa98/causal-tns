"""Evaluate, marginalize, and sample a three-variable causal chain."""

import torch
import torch.nn.functional as nnf

from models import CausalChain


def main() -> None:
    """Run the basic causal-chain example."""
    torch.manual_seed(0)
    chain = CausalChain(
        n_features=3,
        phys_dim=2,
        bond_dim=4,
        directions="rl",
    )

    values = torch.tensor([[0, 1, 0]])
    input = nnf.one_hot(values, num_classes=2).to(torch.get_default_dtype())
    probability = chain.evaluate(input)
    marginal = chain.evaluate([input[:, 0], input[:, 2]], marg_features=[1])
    samples = chain.sample(10, generator=torch.Generator().manual_seed(0))

    print("P(x0=0, x1=1, x2=0):", probability.item())
    print("P(x0=0, x2=0):", marginal.item())
    print("Samples:")
    print(samples)


if __name__ == "__main__":
    main()
