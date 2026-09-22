"""Base class for causal tensor networks."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import tensorkrowch as tk
import torch
import torch.nn.functional as nnf

from causal_node import CausalNode


class CausalTN(tk.TensorNetwork):
    """Base TensorKrowch network that stores ordered causal nodes."""

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name=name)
        self.causal_nodes: OrderedDict[str, CausalNode] = OrderedDict()

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
        """Reset resultants while preserving the static marginalization view."""
        super().reset()
        # TensorKrowch intentionally skips persistent virtual nodes. Clear their
        # operation caches so they cannot retain deleted resultant children.
        for node in self.virtual_nodes.values():
            node._successors = {}

    def set_data_nodes(self) -> None:
        """Create the data nodes required by every causal variable."""
        for node in self.causal_nodes.values():
            node.set_data_nodes()

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

    def _contract_unnormalized(
        self,
        input: torch.Tensor | Sequence[torch.Tensor] | None,
        marg_features: tuple[int, ...],
    ) -> torch.Tensor:
        self.reset()
        inputs = self._prepare_inputs(input, marg_features)
        active_nodes: list[tk.Node] = []
        for i, node in enumerate(self.causal_nodes.values()):
            factors = (
                node.marginalize() if i in marg_features else node.evaluate(inputs[i])
            )
            active_nodes.extend(factors)
        return self.contract(active_nodes)

    def evaluate(
        self,
        input: torch.Tensor | Sequence[torch.Tensor] | None = None,
        marg_features: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Evaluate a normalized probability or partial marginal."""
        marg_features = self._check_marg_features(marg_features)
        numerator = self._contract_unnormalized(input, marg_features)
        return numerator / self.normalize()

    def normalize(self) -> torch.Tensor:
        """Return the global normalization factor."""
        return self._contract_unnormalized(None, tuple(range(self.n_features)))

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
                weights = self._contract_unnormalized(
                    inputs,
                    tuple(range(1, self.n_features)),
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
            weights = self._contract_unnormalized(
                repeated_inputs,
                tuple(range(feature + 1, self.n_features)),
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

    def contract(self, nodes: Sequence[tk.Node]) -> torch.Tensor:
        """Contract model-specific causal factors into a tensor."""
        raise NotImplementedError

    def forward(
        self,
        input: torch.Tensor | Sequence[torch.Tensor] | None = None,
        marg_features: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Call :meth:`evaluate`."""
        return self.evaluate(input=input, marg_features=marg_features)
