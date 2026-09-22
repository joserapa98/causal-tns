# Causal Tensor Networks

This repository implements classical causal tensor networks on top of
[TensorKrowch](https://github.com/joserapa98/tensorkrowch). The mathematical
motivation is described in
[`docs/causal_tn_notes.tex`](docs/causal_tn_notes.tex).

The current implementation provides one-dimensional causal chains and general
directed acyclic graphs. Every variable has a trainable mechanism `K` and one
outgoing channel `B` per child. Their entries are squared during contraction;
`K` is normalized over its physical value and each `B` over its bond. These
local constraints make the complete distribution normalized.

A query plan retains the observed variables and their ancestors. It contains
one parameter-sharing view per retained variable and only the bonds needed for
that query. Compiling a plan creates its fixed TensorKrowch nodes; evaluating
the plan reuses those nodes without changing their connections.

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

# Reuse a compiled query during training or repeated inference.
plan = chain.compile_query(marg_features=[1])
marginal = chain.evaluate(partial_input, plan=plan)

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

`chain.normalize()` returns one, the total mass guaranteed by local
normalization. `chain.evaluate(...)` returns probabilities directly.

## Main classes

- `CausalNode` owns the trainable `K` and outgoing `B` components and creates
  parameter-sharing query views.
- `QueryPlan` stores active features, bonds, and views for one marginal query.
- `CausalTN` stores the ordered causal variables, compiles query plans, and
  defines the common evaluation and sampling interface.
- `CausalChain` builds arbitrary one-dimensional arrow patterns, contracts the
  selected views and samples from left to right.
- `CausalDAG` constructs an arbitrary directed acyclic graph from its adjacency
  matrix and contracts the selected factors with one `einsum` operation.

## General DAG

`adjacency[i][j]` is the bond dimension of the arrow `i -> j`; zero means
there is no arrow. All nonzero entries must be positive integers.

```python
import torch
import torch.nn.functional as nnf

from models import CausalDAG

adjacency = [
    [0, 4, 4, 0],
    [0, 0, 0, 4],
    [0, 0, 0, 4],
    [0, 0, 0, 0],
]

dag = CausalDAG(adjacency, phys_dim=2)
values = torch.tensor([[0, 1, 0, 1]])
input = nnf.one_hot(values, num_classes=2).to(torch.get_default_dtype())
probabilities = dag.evaluate(input)
marginals = dag.evaluate(
    [input[:, 0], input[:, 3]],
    marg_features=[1, 2],
)
```

The adjacency matrix must be square, non-negative, free of self-edges, and acyclic.
`CausalDAG` validates these conditions during construction. Evaluation,
partial marginalization, normalization, and sequential sampling use the same
interface as `CausalChain`.

The query views use persistent virtual `ParamNode` copies created
with `share_tensor=True`. TensorKrowch 1.1.6 reattaches copied leaf edges, so the
implementation contains one localized compatibility helper that restores the
required logical edge objects through TensorKrowch's private graph API. The
dependency is pinned to version 1.1.6 for this reason.

## Development

```bash
conda run -n causal-tns python -m pytest -q
conda run -n causal-tns ruff check src tests experiments
```

Marginal views use a dense copy tensor whose rank grows with the number of
retained children. Plan compilation omits branches without observed descendants.
