"""Evaluate and marginalize a diamond-shaped causal DAG."""

import torch
import torch.nn.functional as nnf

from models import CausalDAG


def main() -> None:
    """Run the basic causal-DAG example."""
    torch.manual_seed(0)
    adjacency = [
        [0, 3, 3, 0],
        [0, 0, 0, 3],
        [0, 0, 0, 3],
        [0, 0, 0, 0],
    ]
    dag = CausalDAG(adjacency, phys_dim=2)

    values = torch.tensor([[0, 1, 0, 1]])
    input = nnf.one_hot(values, num_classes=2).to(torch.get_default_dtype())
    probability = dag.evaluate(input)
    marginal = dag.evaluate(
        [input[:, 0], input[:, 3]],
        marg_features=[1, 2],
    )

    print("P(x0=0, x1=1, x2=0, x3=1):", probability.item())
    print("P(x0=0, x3=1):", marginal.item())


if __name__ == "__main__":
    main()
