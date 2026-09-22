"""One-dimensional causal tensor network."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import tensorkrowch as tk
import torch

from causal_node import CausalNode
from causal_tn import CausalTN


class CausalChain(CausalTN):
    """One-dimensional causal TN with arbitrary bond directions.

    Parameters
    ----------
    n_features : int
        Number of variables in the chain.
    phys_dim : int or sequence[int]
        Physical dimension of every variable.
    bond_dim : int or sequence[int]
        Dimension of every consecutive bond.
    directions : str or sequence[str], optional
        A compact string of ``"r"`` and ``"l"`` characters, a one-element
        sequence containing that string, or one ``"->"``/``"<-"`` entry per
        consecutive bond. If omitted, all arrows point from left to right.
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
        n_features: int,
        phys_dim: int | Sequence[int],
        bond_dim: int | Sequence[int],
        directions: str | Sequence[str] | None = None,
        n_batches: int = 1,
        init_method: str = "randn",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        name: str | None = None,
        **kwargs: float,
    ) -> None:
        if (
            not isinstance(n_features, int)
            or isinstance(n_features, bool)
            or n_features < 1
        ):
            raise ValueError("`n_features` must be a positive integer")
        if (
            not isinstance(n_batches, int)
            or isinstance(n_batches, bool)
            or n_batches < 0
        ):
            raise ValueError("`n_batches` must be a non-negative integer")

        phys_dims = self._expand_dims(phys_dim, n_features, "phys_dim")
        bond_dims = self._expand_dims(bond_dim, max(n_features - 1, 0), "bond_dim")
        directions = self._parse_directions(directions, n_features - 1)
        if len(directions) != n_features - 1:
            raise ValueError("`directions` must have `n_features - 1` entries")

        super().__init__(name=name)
        self.phys_dims = phys_dims
        self.bond_dims = bond_dims
        self.directions = tuple(directions)
        self.n_batches = n_batches
        self.logical_edges: list[tk.Edge] = []

        in_specs = [OrderedDict() for _ in range(n_features)]
        out_specs = [OrderedDict() for _ in range(n_features)]
        links: list[tuple[int, int]] = []
        for i, (dim, direction) in enumerate(zip(bond_dims, self.directions)):
            parent, child = (i, i + 1) if direction == "->" else (i + 1, i)
            parent_name = self._identifier(parent)
            child_name = self._identifier(child)
            out_specs[parent][child_name] = dim
            in_specs[child][parent_name] = dim
            links.append((parent, child))

        for i in range(n_features):
            identifier = self._identifier(i)
            self.causal_nodes[identifier] = CausalNode(
                identifier=identifier,
                phys_dim=phys_dims[i],
                in_bond_dims=in_specs[i],
                out_bond_dims=out_specs[i],
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
    def _parse_directions(
        directions: str | Sequence[str] | None,
        length: int,
    ) -> tuple[str, ...]:
        if directions is None:
            return ("->",) * length
        if isinstance(directions, str):
            values = tuple(directions)
        elif isinstance(directions, Sequence):
            values = tuple(directions)
            if (
                len(values) == 1
                and isinstance(values[0], str)
                and set(values[0]).issubset({"r", "l"})
            ):
                values = tuple(values[0])
        else:
            raise TypeError("`directions` must be a string or a sequence")

        aliases = {"r": "->", "l": "<-", "->": "->", "<-": "<-"}
        if len(values) != length:
            raise ValueError("`directions` must have `n_features - 1` entries")
        if any(value not in aliases for value in values):
            raise ValueError("Directions must be 'r', 'l', '->', or '<-'")
        return tuple(aliases[value] for value in values)

    def contract(self, nodes: Sequence[tk.Node]) -> torch.Tensor:
        """Contract causal factors along the stored chain bonds.

        Parameters
        ----------
        nodes : sequence[tensorkrowch.Node]
            Factors returned by the selected view of every causal node.

        Returns
        -------
        torch.Tensor
            Unnormalized scalar or batched weights.
        """
        active_nodes = list(nodes)

        # Contract only the stored logical bonds; inherited endpoints are stale
        # by design, while edge identity remains valid for all view combinations.
        for edge in self.logical_edges:
            holders = [
                node
                for node in active_nodes
                if any(item is edge for item in node.edges)
            ]
            if len(holders) != 2:
                raise RuntimeError(
                    "A logical edge must occur in exactly two active contraction factors"
                )
            left, right = holders
            result = left @ right
            active_nodes.remove(left)
            active_nodes.remove(right)
            active_nodes.append(result)

        if not active_nodes:
            parameter = next(self.parameters())
            return torch.ones((), device=parameter.device, dtype=parameter.dtype)

        result = active_nodes[0].tensor
        for node in active_nodes[1:]:
            result = result * node.tensor
        return result
