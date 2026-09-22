"""Causal mechanisms and parameter-sharing query views."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import tensorkrowch as tk
import torch

if TYPE_CHECKING:
    from tensorkrowch.components import AbstractNode, Edge


@dataclass(frozen=True)
class CausalView:
    """The fixed local contraction selected for one query."""

    k_node: tk.ParamNode
    out_nodes: OrderedDict[str, tk.ParamNode]
    data_nodes: OrderedDict[str, tk.Node]
    copy_node: tk.Node | None
    observed: bool

    @property
    def input_edges(self) -> list[Edge]:
        """Return physical edges in mechanism-then-child order."""
        if self.copy_node is not None:
            return [self.copy_node["input"]]
        return [
            self.k_node["input"],
            *(node["input"] for node in self.out_nodes.values()),
        ]

    @staticmethod
    def _stochastic(node: tk.ParamNode, axis: str) -> tk.Node:
        """Square and normalize a component, including zero parameter rows."""
        squared = node * node
        positive = squared + torch.finfo(squared.tensor.dtype).tiny
        return tk.renormalize(positive, p=1, axis=axis)

    def contract(self, input: torch.Tensor | None = None) -> list[tk.Node]:
        """Contract the physical value while keeping logical bonds open."""
        if self.observed:
            if input is None:
                raise ValueError("An observed view requires `input`")
            for data in self.data_nodes.values():
                data.set_tensor(input)
            k = self._stochastic(self.k_node, "input")
            factors = [k @ self.data_nodes["k"]]
            for name, node in self.out_nodes.items():
                b = self._stochastic(node, "bond")
                factors.append(b @ self.data_nodes[name])
            return factors

        k = self._stochastic(self.k_node, "input")
        if self.copy_node is None:
            raise RuntimeError("A marginal view needs a copy node")
        for fixed in (self.copy_node, self.data_nodes["copy"]):
            if (
                fixed.tensor.device != k.tensor.device
                or fixed.tensor.dtype != k.tensor.dtype
            ):
                fixed.set_tensor(fixed.tensor.to(k.tensor))
        block = k @ self.copy_node
        block = block @ self.data_nodes["copy"]
        for node in self.out_nodes.values():
            b = self._stochastic(node, "bond")
            block = block @ b
        return [block]


class CausalNode:
    """Store a stochastic mechanism and its outgoing channels.

    `K(x, in_0, ...)` and `B_child(x, bond)` are the only trainable
    components. Query views share their parameters and logical bond edges.
    The complete graph must be connected before creating any views.

    Parameters
    ----------
    identifier : str
        Name of the variable.
    phys_dim : int
        Number of values of the variable.
    in_bond_dims, out_bond_dims : mapping[str, int]
        Bond dimensions indexed by neighbour name.
    network : tensorkrowch.TensorNetwork
        Owner of all nodes and edges.
    n_batches : int
        Number of batch axes on observed inputs.
    init_method, device, dtype, **kwargs
        Passed to the TensorKrowch parameter nodes.
    """

    def __init__(
        self,
        identifier: str,
        phys_dim: int,
        in_bond_dims: Mapping[str, int],
        out_bond_dims: Mapping[str, int],
        network: tk.TensorNetwork,
        n_batches: int = 1,
        init_method: str = "randn",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: float,
    ) -> None:
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("`identifier` must be a non-empty string")
        if not isinstance(phys_dim, int) or isinstance(phys_dim, bool) or phys_dim < 1:
            raise ValueError("`phys_dim` must be a positive integer")
        if (
            not isinstance(n_batches, int)
            or isinstance(n_batches, bool)
            or n_batches < 0
        ):
            raise ValueError("`n_batches` must be a non-negative integer")
        self._validate_bond_dims(in_bond_dims, "in_bond_dims")
        self._validate_bond_dims(out_bond_dims, "out_bond_dims")
        if set(in_bond_dims).intersection(out_bond_dims):
            raise ValueError("A neighbour cannot be both a parent and a child")

        self.identifier = identifier
        self.phys_dim = phys_dim
        self.in_bond_dims = OrderedDict(in_bond_dims)
        self.out_bond_dims = OrderedDict(out_bond_dims)
        self.network = network
        self.n_batches = n_batches
        self.k_node = tk.ParamNode(
            shape=(phys_dim, *self.in_bond_dims.values()),
            axes_names=("input", *(f"in_{name}" for name in self.in_bond_dims)),
            name=f"k_{identifier}",
            network=network,
            init_method=init_method,
            device=device,
            dtype=dtype,
            **kwargs,
        )
        self.out_nodes: OrderedDict[str, tk.ParamNode] = OrderedDict()
        for name, dim in self.out_bond_dims.items():
            self.out_nodes[name] = tk.ParamNode(
                shape=(phys_dim, dim),
                axes_names=("input", "bond"),
                name=f"out_{identifier}_{name}",
                network=network,
                init_method=init_method,
                device=device,
                dtype=dtype,
                **kwargs,
            )

    @staticmethod
    def _validate_bond_dims(dims: Mapping[str, int], arg_name: str) -> None:
        if not isinstance(dims, Mapping):
            raise TypeError(f"`{arg_name}` must be a mapping")
        for name, dim in dims.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"Keys in `{arg_name}` must be non-empty strings")
            if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
                raise ValueError(f"Values in `{arg_name}` must be positive integers")

    @property
    def in_edges(self) -> list[Edge]:
        """Return external incoming bonds in parent order."""
        return [self.k_node[f"in_{name}"] for name in self.in_bond_dims]

    @property
    def out_edges(self) -> list[Edge]:
        """Return external outgoing bonds in child order."""
        return [node["bond"] for node in self.out_nodes.values()]

    @property
    def input_edges(self) -> list[Edge]:
        """Return copied physical edges of the complete base topology."""
        return [
            self.k_node["input"],
            *(node["input"] for node in self.out_nodes.values()),
        ]

    def make_view(self, observed: bool, active_children: Sequence[str]) -> CausalView:
        """Create one static, parameter-sharing view for a query plan."""
        children = tuple(active_children)
        if len(set(children)) != len(children) or any(
            child not in self.out_nodes for child in children
        ):
            raise ValueError("`active_children` must list distinct children")

        k = self._shared_copy(self.k_node)
        self._inherit_edges(
            k, self.k_node, tuple(f"in_{name}" for name in self.in_bond_dims)
        )
        out_nodes: OrderedDict[str, tk.ParamNode] = OrderedDict()
        for name in children:
            out_nodes[name] = self._shared_copy(self.out_nodes[name])
            self._inherit_edges(out_nodes[name], self.out_nodes[name], ("bond",))

        data_nodes: OrderedDict[str, tk.Node] = OrderedDict()
        copy_node = None
        if observed:
            targets = OrderedDict([("k", k), *out_nodes.items()])
            for name, target in targets.items():
                data = self._new_data_node(self.n_batches, ones=False)
                data["feature"] ^ target["input"]
                data_nodes[name] = data
        else:
            axes = ("input", "mechanism", *(f"out_{name}" for name in children))
            copy_node = tk.Node(
                shape=(self.phys_dim,) * len(axes),
                axes_names=axes,
                name=f"copy_{self.identifier}",
                network=self.network,
                virtual=True,
                init_method="copy",
                device=k.tensor.device,
                dtype=k.tensor.dtype,
            )
            copy_node["mechanism"] ^ k["input"]
            for name, node in out_nodes.items():
                copy_node[f"out_{name}"] ^ node["input"]
            data = self._new_data_node(0, ones=True)
            data["feature"] ^ copy_node["input"]
            data_nodes["copy"] = data
        return CausalView(k, out_nodes, data_nodes, copy_node, observed)

    @staticmethod
    def _shared_copy(node: tk.ParamNode) -> tk.ParamNode:
        copy = node.copy(share_tensor=True)
        copy.change_type(virtual=True)
        return copy

    def _inherit_edges(
        self,
        node: AbstractNode,
        node_ref: AbstractNode,
        axes: Sequence[str],
    ) -> None:
        """Attach a view to existing logical edges by identity.

        TensorKrowch copies of non-resultant nodes do not inherit connected
        edges, so the private compatibility operation remains localized here.
        """
        for axis in axes:
            axis_num = node.get_axis_num(axis)
            edge = node_ref._edges[axis_num]
            if node._edges[axis_num] is edge:
                continue
            self.network._remove_edge(node._edges[axis_num])
            node._add_edge(
                edge=edge,
                axis=axis_num,
                node1=node_ref._axes[axis_num].is_node1(),
            )

    def _new_data_node(self, n_batches: int, ones: bool) -> tk.Node:
        shape = (*([1] * n_batches), self.phys_dim)
        axes_names = (*[f"batch_{i}" for i in range(n_batches)], "feature")
        tensor = (
            torch.ones(
                shape,
                device=self.k_node.tensor.device,
                dtype=self.k_node.tensor.dtype,
            )
            if ones
            else None
        )
        return tk.Node(
            shape=None if tensor is not None else shape,
            axes_names=axes_names,
            name=f"data_{self.identifier}",
            network=self.network,
            data=True,
            tensor=tensor,
            init_method=None if tensor is not None else "zeros",
            device=self.k_node.tensor.device,
            dtype=self.k_node.tensor.dtype,
        )
