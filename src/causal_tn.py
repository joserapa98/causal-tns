"""Base class for causal tensor networks."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import tensorkrowch as tk
import torch
import torch.nn.functional as nnf

from causal_node import CausalNode, CausalView


@dataclass(frozen=True)
class QueryPlan:
    """Reusable selection of local views and bonds for one observation mask."""

    network: CausalTN
    marg_features: tuple[int, ...]
    observed_features: tuple[int, ...]
    active_features: tuple[int, ...]
    active_edges: tuple[tk.Edge, ...]
    views: tuple[CausalView, ...]


class CausalTN(tk.TensorNetwork):
    """Base TensorKrowch network that stores ordered causal nodes."""

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name=name)
        self.causal_nodes: OrderedDict[str, CausalNode] = OrderedDict()
        self.query_plans: dict[tuple[int, ...], QueryPlan] = {}
        self.links: list[tuple[int, int]] = []
        self.logical_edge_map: dict[tuple[int, int], tk.Edge] = {}

    @property
    def n_features(self) -> int:
        """Return the number of causal variables."""
        return len(self.causal_nodes)

    @staticmethod
    def _identifier(feature: int) -> str:
        return f"x{feature}"

    @staticmethod
    def _expand_dims(
        dims: int | Sequence[int],
        length: int,
        arg_name: str,
    ) -> tuple[int, ...]:
        if isinstance(dims, int) and not isinstance(dims, bool):
            if dims < 1:
                raise ValueError(f"`{arg_name}` must contain positive integers")
            return (dims,) * length
        if isinstance(dims, (str, bytes)) or not isinstance(dims, Sequence):
            raise TypeError(f"`{arg_name}` must be an integer or a sequence")
        dims = tuple(dims)
        if len(dims) != length:
            raise ValueError(f"`{arg_name}` must have {length} entries")
        if any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim < 1 for dim in dims
        ):
            raise ValueError(f"`{arg_name}` must contain positive integers")
        return dims

    def reset(self) -> None:
        """Reset resultants while preserving compiled query views."""
        super().reset()
        # TensorKrowch intentionally skips persistent virtual nodes. Clear their
        # operation caches so they cannot retain deleted resultant children.
        for node in self.virtual_nodes.values():
            node._successors = {}

    def compile_query(self, marg_features: Sequence[int] | None = None) -> QueryPlan:
        """Compile and cache the views needed for one marginal query.

        Only observed features and their ancestors remain active. Views share
        the base parameters and logical edge objects, so later evaluations do
        not change the graph.
        """
        marg = self._check_marg_features(marg_features)
        if marg in self.query_plans:
            return self.query_plans[marg]
        observed = tuple(i for i in range(self.n_features) if i not in marg)
        active = set(observed)
        parents: dict[int, list[int]] = {i: [] for i in range(self.n_features)}
        for parent, child in self.links:
            parents[child].append(parent)
        stack = list(observed)
        while stack:
            child = stack.pop()
            for parent in parents[child]:
                if parent not in active:
                    active.add(parent)
                    stack.append(parent)

        views: list[CausalView] = []
        active_names = {self._identifier(i) for i in active}
        for i in sorted(active):
            node = self.causal_nodes[self._identifier(i)]
            children = tuple(
                name for name in node.out_bond_dims if name in active_names
            )
            views.append(node.make_view(i in observed, children))
        active_edges = tuple(
            self.logical_edge_map[parent, child]
            for parent, child in self.links
            if parent in active and child in active
        )
        plan = QueryPlan(
            network=self,
            marg_features=marg,
            observed_features=observed,
            active_features=tuple(sorted(active)),
            active_edges=active_edges,
            views=tuple(views),
        )
        self.query_plans[marg] = plan
        return plan

    def _check_marg_features(
        self, marg_features: Sequence[int] | None
    ) -> tuple[int, ...]:
        if marg_features is None:
            return ()
        if isinstance(marg_features, (str, bytes)) or not isinstance(
            marg_features, Sequence
        ):
            raise TypeError("`marg_features` must be a sequence of integers")
        features = tuple(marg_features)
        if any(not isinstance(i, int) or isinstance(i, bool) for i in features):
            raise TypeError("`marg_features` must contain integers")
        if len(set(features)) != len(features):
            raise ValueError("`marg_features` cannot contain duplicates")
        if any(i < 0 or i >= self.n_features for i in features):
            raise ValueError("`marg_features` contains an index outside the network")
        return tuple(sorted(features))

    def _prepare_inputs(
        self,
        input: torch.Tensor | Sequence[torch.Tensor] | None,
        marg_features: tuple[int, ...],
    ) -> dict[int, torch.Tensor]:
        in_features = [i for i in range(self.n_features) if i not in marg_features]
        if not in_features:
            if input is not None and not (
                isinstance(input, Sequence)
                and not isinstance(input, torch.Tensor)
                and len(input) == 0
            ):
                raise ValueError(
                    "`input` must be None when all features are marginalized"
                )
            return {}
        if input is None:
            raise ValueError("`input` is required for non-marginalized features")

        if isinstance(input, torch.Tensor):
            dims = {self.phys_dims[i] for i in in_features}
            if len(dims) != 1:
                raise ValueError(
                    "Use a sequence of tensors for heterogeneous physical dimensions"
                )
            expected_ndim = self.n_batches + 2
            if input.ndim != expected_ndim:
                raise ValueError(
                    f"Expected {expected_ndim} input dimensions, got {input.ndim}"
                )
            if input.shape[-2] != len(in_features):
                raise ValueError(
                    f"Expected {len(in_features)} input features, got {input.shape[-2]}"
                )
            if input.shape[-1] != next(iter(dims)):
                raise ValueError("The last input dimension does not match `phys_dim`")
            return {
                feature: input.select(-2, j) for j, feature in enumerate(in_features)
            }

        if isinstance(input, (str, bytes)) or not isinstance(input, Sequence):
            raise TypeError("`input` must be a tensor or a sequence of tensors")
        if len(input) != len(in_features):
            raise ValueError(
                f"Expected {len(in_features)} input tensors, got {len(input)}"
            )

        result: dict[int, torch.Tensor] = {}
        batch_shape: torch.Size | None = None
        for feature, tensor in zip(in_features, input):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("Every input element must be a torch.Tensor")
            if tensor.ndim != self.n_batches + 1:
                raise ValueError(
                    f"Expected {self.n_batches + 1} dimensions for feature {feature}"
                )
            if tensor.shape[-1] != self.phys_dims[feature]:
                raise ValueError(f"Physical dimension mismatch for feature {feature}")
            if batch_shape is None:
                batch_shape = tensor.shape[:-1]
            elif tensor.shape[:-1] != batch_shape:
                raise ValueError("All input tensors must have the same batch shape")
            result[feature] = tensor
        return result

    def _evaluate_plan(
        self,
        input: torch.Tensor | Sequence[torch.Tensor] | None,
        plan: QueryPlan,
    ) -> torch.Tensor:
        inputs = self._prepare_inputs(input, plan.marg_features)
        self.reset()
        active_nodes: list[tk.Node] = []
        for i, view in zip(plan.active_features, plan.views):
            active_nodes.extend(view.contract(inputs.get(i)))
        return self.contract(active_nodes, plan.active_edges)

    def evaluate(
        self,
        input: torch.Tensor | Sequence[torch.Tensor] | None = None,
        marg_features: Sequence[int] | None = None,
        plan: QueryPlan | None = None,
    ) -> torch.Tensor:
        """Evaluate a probability or partial marginal using a cached plan."""
        if plan is not None:
            if marg_features is not None:
                raise ValueError("Pass either `plan` or `marg_features`")
            if not isinstance(plan, QueryPlan) or plan.network is not self:
                raise ValueError("`plan` must belong to this network")
        else:
            plan = self.compile_query(marg_features)
        return self._evaluate_plan(input, plan)

    def normalize(self) -> torch.Tensor:
        """Return the total mass, fixed to one by local normalization."""
        parameter = next(self.parameters())
        return torch.ones((), device=parameter.device, dtype=parameter.dtype)

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw samples through sequential partial marginalization."""
        if (
            not isinstance(n_samples, int)
            or isinstance(n_samples, bool)
            or n_samples < 1
        ):
            raise ValueError("`n_samples` must be a positive integer")
        if self.n_batches != 1:
            raise RuntimeError("Sampling requires `n_batches=1`")
        parameter = next(self.parameters())
        samples = torch.empty(
            (n_samples, self.n_features),
            device=parameter.device,
            dtype=torch.long,
        )

        for feature, dim in enumerate(self.phys_dims):
            if feature == 0:
                inputs = [
                    torch.eye(dim, device=parameter.device, dtype=parameter.dtype)
                ]
                weights = self.evaluate(
                    inputs,
                    plan=self.compile_query(tuple(range(1, self.n_features))),
                ).reshape(1, dim)
                probs = self._normalize_rows(weights)
                samples[:, feature] = torch.multinomial(
                    probs[0],
                    n_samples,
                    replacement=True,
                    generator=generator,
                )
                continue

            repeated_inputs: list[torch.Tensor] = []
            for previous in range(feature):
                values = nnf.one_hot(
                    samples[:, previous],
                    num_classes=self.phys_dims[previous],
                ).to(dtype=parameter.dtype)
                repeated_inputs.append(values.repeat_interleave(dim, dim=0))
            repeated_inputs.append(
                torch.eye(dim, device=parameter.device, dtype=parameter.dtype).repeat(
                    n_samples, 1
                )
            )
            weights = self.evaluate(
                repeated_inputs,
                plan=self.compile_query(tuple(range(feature + 1, self.n_features))),
            ).reshape(n_samples, dim)
            probs = self._normalize_rows(weights)
            samples[:, feature] = torch.multinomial(
                probs,
                1,
                replacement=True,
                generator=generator,
            ).squeeze(-1)
        return samples

    @staticmethod
    def _normalize_rows(weights: torch.Tensor) -> torch.Tensor:
        weights = weights.clamp_min(0)
        totals = weights.sum(dim=-1, keepdim=True)
        if torch.any(totals <= 0):
            raise RuntimeError("Cannot sample from zero conditional weights")
        return weights / totals

    def contract(
        self, nodes: Sequence[tk.Node], edges: Sequence[tk.Edge] | None = None
    ) -> torch.Tensor:
        """Contract model-specific causal factors into a tensor."""
        raise NotImplementedError

    def forward(
        self,
        input: torch.Tensor | Sequence[torch.Tensor] | None = None,
        marg_features: Sequence[int] | None = None,
        plan: QueryPlan | None = None,
    ) -> torch.Tensor:
        """Call :meth:`evaluate`."""
        return self.evaluate(input=input, marg_features=marg_features, plan=plan)
