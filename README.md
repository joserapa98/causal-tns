# Causal Tensor Networks

This repository implements classical causal tensor networks on top of
[TensorKrowch](https://github.com/joserapa98/tensorkrowch). The mathematical
motivation is described in
[`docs/causal_tn_notes.tex`](docs/causal_tn_notes.tex).

The current implementation provides one-dimensional causal chains and general
directed acyclic graphs. Every variable is represented by a structured
`CausalNode` with two static views:

- the `eval` view uses explicit copies of the physical input;
- the `marg` view uses a fixed copy tensor and shares all trainable tensors with
  the `eval` view.

All trainable components are squared during contraction. The local mechanism
`K` is normalized over its physical input, while the complete network uses a
global normalization factor.

## Environment

```bash
conda env create -f environment.yml
conda activate causal-tns
```

To update dependencies after changing `environment.yml`:

```bash
conda env update -n causal-tns -f environment.yml --prune
```

Reusable code belongs in `src/`, and experiments belong in `experiments/`.

## Basic chain

```python
import torch
import torch.nn.functional as nnf

from models import CausalChain

chain = CausalChain(
    n_features=3,
    phys_dim=2,
    bond_dim=4,
    directions="rl",  # x0 -> x1 <- x2
)

# Evaluate P(x0=0, x1=1, x2=0) for a batch of one assignment.
values = torch.tensor([[0, 1, 0]])
input = nnf.one_hot(values, num_classes=2).to(torch.get_default_dtype())
probability = chain.evaluate(input)

# Evaluate P(x0=0, x2=0) by marginalizing x1.
partial_input = [input[:, 0], input[:, 2]]
marginal = chain.evaluate(partial_input, marg_features=[1])

# Draw complete assignments.
samples = chain.sample(100)
```

For equal physical dimensions, `evaluate` accepts one tensor with shape
`(*batch, n_inputs, phys_dim)`. For unequal physical dimensions, pass one
`(*batch, phys_dim_i)` tensor per non-marginalized feature. These tensors are
ordered by feature index. Use `input=None` when every feature is marginalized.

For chains, directions can use arrow tokens or a compact string. The following
forms are equivalent:

```python
directions = ["->", "<-", "->"]
directions = ["r", "l", "r"]
directions = "rlr"
directions = ["rlr"]
```

`chain.normalize()` returns the unnormalized mass obtained by marginalizing all
variables. `chain.evaluate(...)` divides each requested weight by this value.

## Main classes

- `CausalNode` owns the trainable `K`, incoming `A`, and outgoing `B`
  components. It returns disconnected factors from `evaluate` and
  `marginalize`.
- `CausalTN` stores the ordered causal variables and defines the common query
  interface.
- `CausalChain` builds arbitrary one-dimensional arrow patterns, contracts the
  selected views, normalizes probabilities, and samples from left to right.
- `CausalDAG` constructs an arbitrary directed acyclic graph from its adjacency
  matrix and contracts the selected factors with one `einsum` operation.

## General DAG

`adjacency[i][j] = 1` represents the arrow `i -> j`. `bond_dim` may be one
shared integer or a square matrix specifying the dimension of each active
edge.

```python
import torch
import torch.nn.functional as nnf

from models import CausalDAG

adjacency = [
    [0, 1, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 0],
]

dag = CausalDAG(adjacency, phys_dim=2, bond_dim=4)
values = torch.tensor([[0, 1, 0, 1]])
input = nnf.one_hot(values, num_classes=2).to(torch.get_default_dtype())
probabilities = dag.evaluate(input)
marginals = dag.evaluate(
    [input[:, 0], input[:, 3]],
    marg_features=[1, 2],
)
```

The adjacency matrix must be square, binary, free of self-edges, and acyclic.
`CausalDAG` validates these conditions during construction. Evaluation,
partial marginalization, normalization, and sequential sampling use the same
interface as `CausalChain`.

The marginalization nodes are persistent virtual `ParamNode` copies created
with `share_tensor=True`. TensorKrowch 1.1.6 reattaches copied leaf edges, so the
implementation contains one localized compatibility helper that restores the
required logical edge objects through TensorKrowch's private graph API. The
dependency is pinned to version 1.1.6 for this reason.

## Development

```bash
conda run -n causal-tns python -m pytest -q
conda run -n causal-tns ruff check src tests experiments
```

The marginalization copy tensor is dense. Its rank grows with the out-degree of
a causal variable, so the current representation is intended first for chains
and other low-degree networks.
