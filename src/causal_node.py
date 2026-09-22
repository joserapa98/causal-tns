"""Structured causal nodes with evaluation and marginalization views."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import tensorkrowch as tk
import torch

if TYPE_CHECKING:
    from tensorkrowch.components import AbstractNode, Edge


class CausalNode:
    """Represent one classical variable through structured TN components.

    Parameters
    ----------
    identifier : str
        Identifier used to name the components and neighbouring variables.
    phys_dim : int
        Number of possible values of the variable.
    in_bond_dims : mapping[str, int]
        Bond dimensions indexed by parent identifier.
    out_bond_dims : mapping[str, int]
        Bond dimensions indexed by child identifier.
    network : tensorkrowch.TensorNetwork
        Network that owns all the internal nodes.
    n_batches : int
        Number of batch axes used by evaluation data nodes.
    init_method : str
        TensorKrowch initialization method for trainable nodes.
    device : torch.device, optional
        Device of trainable and fixed tensors.
    dtype : torch.dtype, optional
        Data type of trainable and fixed tensors.
    **kwargs
        Extra arguments passed to ``tk.ParamNode`` initialization.

    Notes
    -----
    The evaluation topology must be connected to neighbouring causal nodes
    before :meth:`make_marginal_view` is called.
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
        overlap = set(in_bond_dims).intersection(out_bond_dims)
        if overlap:
            raise ValueError(
                f"A neighbour cannot be both a parent and a child: {sorted(overlap)}"
            )

        self.identifier = identifier
        self.phys_dim = phys_dim
        self.in_bond_dims = OrderedDict(in_bond_dims)
        self.out_bond_dims = OrderedDict(out_bond_dims)
        self.network = network
        self.n_batches = n_batches
        self._device = device
        self._dtype = dtype
        self._marg_view_ready = False
        self._data_nodes_ready = False

        in_axes = [f"in_{name}" for name in self.in_bond_dims]
        self.eval_k_node = tk.ParamNode(
            shape=(phys_dim, *self.in_bond_dims.values()),
            axes_names=("input", *in_axes),
            name=f"eval_k_{identifier}",
            network=network,
            init_method=init_method,
            device=device,
            dtype=dtype,
            **kwargs,
        )

        self.eval_in_nodes: OrderedDict[str, tk.ParamNode] = OrderedDict()
        for name, dim in self.in_bond_dims.items():
            node = tk.ParamNode(
                shape=(dim, dim),
                axes_names=("bond", "mechanism"),
                name=f"eval_in_{name}_{identifier}",
                network=network,
                init_method=init_method,
                device=device,
                dtype=dtype,
                **kwargs,
            )
            node["mechanism"] ^ self.eval_k_node[f"in_{name}"]
            self.eval_in_nodes[name] = node

        self.eval_out_nodes: OrderedDict[str, tk.ParamNode] = OrderedDict()
        for name, dim in self.out_bond_dims.items():
            self.eval_out_nodes[name] = tk.ParamNode(
                shape=(phys_dim, dim),
                axes_names=("input", "bond"),
                name=f"eval_out_{identifier}_{name}",
                network=network,
                init_method=init_method,
                device=device,
                dtype=dtype,
                **kwargs,
            )

        self.marg_k_node: tk.ParamNode | None = None
        self.marg_in_nodes: OrderedDict[str, tk.ParamNode] = OrderedDict()
        self.marg_out_nodes: OrderedDict[str, tk.ParamNode] = OrderedDict()
        self.marg_copy_node: tk.Node | None = None
        self.eval_data_nodes: OrderedDict[str, tk.Node] = OrderedDict()
        self.marg_data_nodes: OrderedDict[str, tk.Node] = OrderedDict()

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
        """Return external incoming edges in parent order."""
        return [node["bond"] for node in self.eval_in_nodes.values()]

    @property
    def out_edges(self) -> list[Edge]:
        """Return external outgoing edges in child order."""
        return [node["bond"] for node in self.eval_out_nodes.values()]

    @property
    def eval_input_edges(self) -> list[Edge]:
        """Return copied physical edges used for evaluation."""
        return [
            self.eval_k_node["input"],
            *(node["input"] for node in self.eval_out_nodes.values()),
        ]

    @property
    def input_edges(self) -> list[Edge]:
        """Return the physical input edges of the evaluation view."""
        return self.eval_input_edges

    @property
    def marg_input_edges(self) -> list[Edge]:
        """Return physical edges connected to marginalization data."""
        if not self._marg_view_ready:
            raise RuntimeError("The marginalization view has not been created")
        if self.marg_copy_node is not None:
            return [self.marg_copy_node["input"]]
        return [node["mechanism"] for node in self.marg_in_nodes.values()]

    def make_marginal_view(self) -> None:
        """Create parameter-sharing nodes for marginalization.

        This method must be called once, after all external evaluation bonds
        have been connected.
        """
        if self._marg_view_ready:
            raise RuntimeError("The marginalization view already exists")

        if self.eval_out_nodes:
            self._make_copied_marg_view()
        else:
            self._make_normalized_leaf_marg_view()
        self._marg_view_ready = True

    def _make_copied_marg_view(self) -> None:
        self.marg_k_node = self._shared_copy(self.eval_k_node)
        for name, node in self.eval_in_nodes.items():
            self.marg_in_nodes[name] = self._shared_copy(node)
        for name, node in self.eval_out_nodes.items():
            self.marg_out_nodes[name] = self._shared_copy(node)

        # Restore internal mechanism edges and every external logical bond.
        for name, node in self.marg_in_nodes.items():
            self._inherit_edges(node, self.eval_in_nodes[name], ("bond", "mechanism"))
        self._inherit_edges(
            self.marg_k_node,
            self.eval_k_node,
            tuple(f"in_{name}" for name in self.in_bond_dims),
        )
        for name, node in self.marg_out_nodes.items():
            self._inherit_edges(node, self.eval_out_nodes[name], ("bond",))

        copy_axes = (
            "input",
            "mechanism",
            *[f"out_{name}" for name in self.out_bond_dims],
        )
        self.marg_copy_node = tk.Node(
            shape=(self.phys_dim,) * len(copy_axes),
            axes_names=copy_axes,
            name=f"marg_copy_{self.identifier}",
            network=self.network,
            virtual=True,
            init_method="copy",
            device=self._device,
            dtype=self._dtype,
        )
        self.marg_copy_node["mechanism"] ^ self.marg_k_node["input"]
        for name, node in self.marg_out_nodes.items():
            self.marg_copy_node[f"out_{name}"] ^ node["input"]

    def _make_normalized_leaf_marg_view(self) -> None:
        for name, eval_node in self.eval_in_nodes.items():
            node = self._shared_copy(eval_node)
            self._inherit_edges(node, eval_node, ("bond",))
            self._make_dangling_edge(node, eval_node, "mechanism")
            self.marg_in_nodes[name] = node

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
        """Restore exact edge objects after a sharing copy.

        TensorKrowch reattaches the edges of non-resultant copied nodes. This
        compatibility operation intentionally uses private graph primitives in
        one isolated place.
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

    def _make_dangling_edge(
        self,
        node: AbstractNode,
        node_ref: AbstractNode,
        axis: str,
    ) -> None:
        axis_num = node.get_axis_num(axis)
        if node._edges[axis_num] is not node_ref._edges[axis_num]:
            self.network._remove_edge(node._edges[axis_num])
        edge = tk.Edge(node1=node, axis1=axis_num)
        node._add_edge(edge=edge, axis=axis_num, node1=True)
        self.network._add_edge(edge)

    def set_data_nodes(self) -> None:
        """Create evaluation and marginalization data nodes."""
        if not self._marg_view_ready:
            raise RuntimeError("Create the marginalization view first")
        if self._data_nodes_ready:
            raise RuntimeError("Data nodes already exist")

        eval_targets = OrderedDict([("k", self.eval_k_node)])
        eval_targets.update(
            (f"out_{name}", node) for name, node in self.eval_out_nodes.items()
        )
        for label, target in eval_targets.items():
            data = self._new_data_node(
                size=self.phys_dim,
                name=f"eval_data_{self.identifier}_{label}",
                n_batches=self.n_batches,
            )
            data["feature"] ^ target["input"]
            self.eval_data_nodes[label] = data

        if self.marg_copy_node is not None:
            data = self._new_data_node(
                size=self.phys_dim,
                name=f"marg_data_{self.identifier}",
                n_batches=0,
                ones=True,
            )
            data["feature"] ^ self.marg_copy_node["input"]
            self.marg_data_nodes["copy"] = data
        else:
            for name, node in self.marg_in_nodes.items():
                data = self._new_data_node(
                    size=self.in_bond_dims[name],
                    name=f"marg_data_{self.identifier}_{name}",
                    n_batches=0,
                    ones=True,
                )
                data["feature"] ^ node["mechanism"]
                self.marg_data_nodes[name] = data

        self._data_nodes_ready = True

    def _new_data_node(
        self,
        size: int,
        name: str,
        n_batches: int,
        ones: bool = False,
    ) -> tk.Node:
        shape = (*([1] * n_batches), size)
        axes_names = (*[f"batch_{i}" for i in range(n_batches)], "feature")
        tensor = None
        if ones:
            tensor = torch.ones(shape, device=self._device, dtype=self._dtype)
        return tk.Node(
            shape=None if tensor is not None else shape,
            axes_names=axes_names,
            name=name,
            network=self.network,
            data=True,
            tensor=tensor,
            init_method=None if tensor is not None else "zeros",
            device=self._device,
            dtype=self._dtype,
        )

    def evaluate(self, input: torch.Tensor) -> list[tk.Node]:
        """Evaluate the variable and return its disconnected factors.

        Parameters
        ----------
        input : torch.Tensor
            Canonical or relaxed input with shape ``(*batch, phys_dim)``.

        Returns
        -------
        list[tensorkrowch.Node]
            Incoming block followed by one factor per outgoing bond.
        """
        self._check_ready()
        if not isinstance(input, torch.Tensor):
            raise TypeError("`input` must be a torch.Tensor")
        if input.ndim != self.n_batches + 1:
            raise ValueError(
                f"Expected {self.n_batches + 1} input dimensions, got {input.ndim}"
            )
        if input.shape[-1] != self.phys_dim:
            raise ValueError(
                f"Expected physical dimension {self.phys_dim}, got {input.shape[-1]}"
            )

        for data in self.eval_data_nodes.values():
            data.set_tensor(input)

        k_node = tk.renormalize(self.eval_k_node * self.eval_k_node, p=1, axis="input")
        in_block = k_node @ self.eval_data_nodes["k"]
        for node in self.eval_in_nodes.values():
            in_block = in_block @ (node * node)

        factors = [in_block]
        for name, node in self.eval_out_nodes.items():
            factors.append((node * node) @ self.eval_data_nodes[f"out_{name}"])
        return factors

    def marginalize(self) -> list[tk.Node]:
        """Marginalize the variable and return its contraction factors."""
        self._check_ready()
        if self.marg_copy_node is None:
            return [
                (node * node) @ self.marg_data_nodes[name]
                for name, node in self.marg_in_nodes.items()
            ]

        if self.marg_k_node is None:
            raise RuntimeError("Invalid marginalization view")
        k_node = tk.renormalize(self.marg_k_node * self.marg_k_node, p=1, axis="input")
        block = k_node
        for node in self.marg_in_nodes.values():
            block = block @ (node * node)

        block = block @ self.marg_copy_node
        block = block @ self.marg_data_nodes["copy"]
        for node in self.marg_out_nodes.values():
            block = block @ (node * node)
        return [block]

    def _check_ready(self) -> None:
        if not self._marg_view_ready or not self._data_nodes_ready:
            raise RuntimeError("The causal node has not been fully constructed")
