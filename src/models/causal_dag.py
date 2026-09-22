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
        Square matrix of non-negative integers. ``adjacency[i][j]`` is the
        bond dimension of the directed edge ``i -> j``; zero means no edge.
    phys_dim : int or sequence[int]
        Physical dimension of every variable.
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

        topological_order = self._topological_order(adjacency_tensor != 0)
        phys_dims = self._expand_dims(phys_dim, n_features, "phys_dim")
        bond_dims = tuple(tuple(row) for row in adjacency_tensor.tolist())

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
                parent_node.out_nodes[self._identifier(child)]["bond"]
                ^ child_node.k_node[f"in_{self._identifier(parent)}"]
            )
            self.logical_edges.append(edge)
            self.logical_edge_map[parent, child] = edge
        self.links = links

    @staticmethod
    def _check_adjacency(
        adjacency: torch.Tensor | Sequence[Sequence[int]],
    ) -> torch.Tensor:
        try:
            tensor = torch.as_tensor(adjacency)
        except (TypeError, ValueError, RuntimeError) as err:
            raise TypeError("`adjacency` must be a square integer matrix") from err
        if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
            raise ValueError("`adjacency` must be a square matrix")
        if tensor.shape[0] < 1:
            raise ValueError("`adjacency` must contain at least one variable")
        if (
            tensor.dtype == torch.bool
            or torch.is_floating_point(tensor)
            or tensor.is_complex()
        ):
            raise TypeError("`adjacency` must contain integer bond dimensions")
        if torch.any(torch.diag(tensor) != 0):
            raise ValueError("`adjacency` cannot contain self-edges")
        if torch.any(tensor < 0):
            raise ValueError("`adjacency` cannot contain negative bond dimensions")
        return tensor.to(dtype=torch.int64, device="cpu").clone()

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

    def contract(
        self, nodes: Sequence[tk.Node], edges: Sequence[tk.Edge] | None = None
    ) -> torch.Tensor:
        """Contract all selected causal factors in one einsum operation."""
        nodes = list(nodes)
        if edges is None:
            edges = self.logical_edges
        if not nodes:
            parameter = next(self.parameters())
            return torch.ones((), device=parameter.device, dtype=parameter.dtype)

        edge_symbols = {id(edge): get_symbol(i) for i, edge in enumerate(edges)}
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
