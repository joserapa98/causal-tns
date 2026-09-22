"""Causal tensor network for a directed acyclic graph."""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Sequence

import tensorkrowch as tk
import torch
from opt_einsum import get_symbol

from causal_node import CausalNode
from causal_tn import CausalTN


class CausalDAG(CausalTN):
    """Causal TN defined by an adjacency matrix.

    Parameters
    ----------
    adjacency : torch.Tensor or sequence[sequence[int]]
        Square binary matrix. A nonzero entry ``adjacency[i][j]`` represents
        the directed edge ``i -> j``.
    phys_dim : int or sequence[int]
        Physical dimension of every variable.
    bond_dim : int or sequence[sequence[int]]
        Shared bond dimension or a square matrix with one positive dimension
        for every active adjacency entry.
    n_batches : int
        Number of batch axes in evaluation inputs.
    init_method : str
        TensorKrowch initialization method for trainable components.
    device : torch.device, optional
        Device of the network tensors.
    dtype : torch.dtype, optional
        Data type of the network tensors.
    name : str, optional
        TensorKrowch network name.
    **kwargs
        Extra arguments passed to trainable node initialization.
    """

    def __init__(
        self,
        adjacency: torch.Tensor | Sequence[Sequence[int]],
        phys_dim: int | Sequence[int],
        bond_dim: int | Sequence[Sequence[int]],
        n_batches: int = 1,
        init_method: str = "randn",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        name: str | None = None,
        **kwargs: float,
    ) -> None:
        adjacency_tensor = self._check_adjacency(adjacency)
        n_features = adjacency_tensor.shape[0]
        if (
            not isinstance(n_batches, int)
            or isinstance(n_batches, bool)
            or n_batches < 0
        ):
            raise ValueError("`n_batches` must be a non-negative integer")

        topological_order = self._topological_order(adjacency_tensor)
        phys_dims = self._expand_dims(phys_dim, n_features, "phys_dim")
        bond_dims = self._check_bond_dims(bond_dim, adjacency_tensor)

        super().__init__(name=name)
        self.adjacency = adjacency_tensor
        self.phys_dims = phys_dims
        self.bond_dims = bond_dims
        self.n_batches = n_batches
        self.topological_order = topological_order
        self.logical_edges: list[tk.Edge] = []

        in_specs = [OrderedDict() for _ in range(n_features)]
        out_specs = [OrderedDict() for _ in range(n_features)]
        links: list[tuple[int, int]] = []
        for parent in range(n_features):
            for child in range(n_features):
                if not adjacency_tensor[parent, child]:
                    continue
                dim = bond_dims[parent][child]
                out_specs[parent][self._identifier(child)] = dim
                in_specs[child][self._identifier(parent)] = dim
                links.append((parent, child))

        for feature in range(n_features):
            identifier = self._identifier(feature)
            self.causal_nodes[identifier] = CausalNode(
                identifier=identifier,
                phys_dim=phys_dims[feature],
                in_bond_dims=in_specs[feature],
                out_bond_dims=out_specs[feature],
                network=self,
                n_batches=n_batches,
                init_method=init_method,
                device=device,
                dtype=dtype,
                **kwargs,
            )

        for parent, child in links:
            parent_node = self.causal_nodes[self._identifier(parent)]
            child_node = self.causal_nodes[self._identifier(child)]
            edge = (
                parent_node.eval_out_nodes[self._identifier(child)]["bond"]
                ^ child_node.eval_in_nodes[self._identifier(parent)]["bond"]
            )
            self.logical_edges.append(edge)

        for node in self.causal_nodes.values():
            node.make_marginal_view()
        self.set_data_nodes()

    @staticmethod
    def _check_adjacency(
        adjacency: torch.Tensor | Sequence[Sequence[int]],
    ) -> torch.Tensor:
        try:
            tensor = torch.as_tensor(adjacency)
        except (TypeError, ValueError, RuntimeError) as err:
            raise TypeError("`adjacency` must be a square binary matrix") from err
        if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
            raise ValueError("`adjacency` must be a square matrix")
        if tensor.shape[0] < 1:
            raise ValueError("`adjacency` must contain at least one variable")
        if torch.any((tensor != 0) & (tensor != 1)):
            raise ValueError("`adjacency` must contain only zeros and ones")
        if torch.any(torch.diag(tensor) != 0):
            raise ValueError("`adjacency` cannot contain self-edges")
        return tensor.to(dtype=torch.bool, device="cpu")

    @staticmethod
    def _topological_order(adjacency: torch.Tensor) -> tuple[int, ...]:
        n_features = adjacency.shape[0]
        in_degree = adjacency.sum(dim=0).tolist()
        queue = deque(i for i, degree in enumerate(in_degree) if degree == 0)
        order: list[int] = []
        while queue:
            parent = queue.popleft()
            order.append(parent)
            for child in torch.nonzero(adjacency[parent], as_tuple=False).flatten():
                child = int(child)
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        if len(order) != n_features:
            raise ValueError("`adjacency` must define a directed acyclic graph")
        return tuple(order)

    @staticmethod
    def _check_bond_dims(
        bond_dim: int | Sequence[Sequence[int]],
        adjacency: torch.Tensor,
    ) -> tuple[tuple[int, ...], ...]:
        n_features = adjacency.shape[0]
        if isinstance(bond_dim, int) and not isinstance(bond_dim, bool):
            if bond_dim < 1:
                raise ValueError("`bond_dim` must be positive")
            return tuple((bond_dim,) * n_features for _ in range(n_features))
        try:
            matrix = torch.as_tensor(bond_dim)
        except (TypeError, ValueError, RuntimeError) as err:
            raise TypeError("`bond_dim` must be an integer or a square matrix") from err
        if matrix.shape != adjacency.shape:
            raise ValueError("A bond-dimension matrix must match `adjacency`")

        result: list[tuple[int, ...]] = []
        for parent in range(n_features):
            row: list[int] = []
            for child in range(n_features):
                value = matrix[parent, child].item()
                if adjacency[parent, child] and (
                    isinstance(value, bool)
                    or not float(value).is_integer()
                    or value < 1
                ):
                    raise ValueError("Active bond dimensions must be positive integers")
                row.append(int(value) if adjacency[parent, child] else 0)
            result.append(tuple(row))
        return tuple(result)

    def contract(self, nodes: Sequence[tk.Node]) -> torch.Tensor:
        """Contract all selected causal factors in one einsum operation."""
        nodes = list(nodes)
        if not nodes:
            parameter = next(self.parameters())
            return torch.ones((), device=parameter.device, dtype=parameter.dtype)

        edge_symbols = {
            id(edge): get_symbol(i) for i, edge in enumerate(self.logical_edges)
        }
        edge_counts = dict.fromkeys(edge_symbols, 0)
        batch_symbols: dict[str, str] = {}
        next_symbol = len(edge_symbols)
        terms: list[str] = []

        for node in nodes:
            term = ""
            for axis, edge in zip(node.axes_names, node.edges):
                edge_id = id(edge)
                if edge_id in edge_symbols:
                    symbol = edge_symbols[edge_id]
                    edge_counts[edge_id] += 1
                elif axis.startswith("batch"):
                    if axis not in batch_symbols:
                        batch_symbols[axis] = get_symbol(next_symbol)
                        next_symbol += 1
                    symbol = batch_symbols[axis]
                else:
                    raise RuntimeError(f"Unexpected open edge `{axis}` in DAG factor")
                term += symbol
            terms.append(term)

        if any(count != 2 for count in edge_counts.values()):
            raise RuntimeError(
                "Every logical DAG edge must occur in exactly two factors"
            )

        def batch_order(axis: str) -> int:
            if axis == "batch":
                return 0
            return int(axis.rsplit("_", maxsplit=1)[-1])

        output = "".join(
            batch_symbols[name] for name in sorted(batch_symbols, key=batch_order)
        )
        equation = ",".join(terms) + "->" + output
        return tk.einsum(equation, *nodes).tensor
