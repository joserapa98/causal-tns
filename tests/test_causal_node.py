"""Tests for the structured causal node views."""

import torch

from models import CausalChain


def test_marg_nodes_share_parameters_and_logical_edges() -> None:
    chain = CausalChain(
        n_features=2,
        phys_dim=2,
        bond_dim=3,
        directions=["->"],
    )
    parent, child = chain.causal_nodes.values()

    assert parent.marg_k_node is not None
    assert parent.marg_k_node.tensor is parent.eval_k_node.tensor
    assert parent.marg_out_nodes["x1"].tensor is parent.eval_out_nodes["x1"].tensor
    assert child.marg_in_nodes["x0"].tensor is child.eval_in_nodes["x0"].tensor

    edge = chain.logical_edges[0]
    assert parent.eval_out_nodes["x1"]["bond"] is edge
    assert parent.marg_out_nodes["x1"]["bond"] is edge
    assert child.eval_in_nodes["x0"]["bond"] is edge
    assert child.marg_in_nodes["x0"]["bond"] is edge


def test_all_mixed_views_contract_and_differentiate() -> None:
    chain = CausalChain(
        n_features=2,
        phys_dim=2,
        bond_dim=3,
        directions=["->"],
        dtype=torch.float64,
    )
    eye = torch.eye(2, dtype=torch.float64)

    full = chain.evaluate(torch.stack((eye, eye), dim=1))
    left_marg = chain.evaluate([eye], marg_features=[0])
    right_marg = chain.evaluate([eye], marg_features=[1])
    norm = chain.normalize()
    loss = full.sum() + left_marg.sum() + right_marg.sum() + norm
    loss.backward()

    assert full.shape == (2,)
    assert left_marg.shape == (2,)
    assert right_marg.shape == (2,)
    assert norm.ndim == 0
    assert all(parameter.grad is not None for parameter in chain.parameters())


def test_reset_preserves_shared_parameter_references() -> None:
    chain = CausalChain(3, 2, 2, directions=["->", "<-"])
    refs = []
    for node in chain.causal_nodes.values():
        if node.marg_k_node is not None:
            refs.append((node.marg_k_node, node.eval_k_node))
        refs.extend(zip(node.marg_in_nodes.values(), node.eval_in_nodes.values()))
        refs.extend(zip(node.marg_out_nodes.values(), node.eval_out_nodes.values()))

    for _ in range(3):
        chain.normalize()
        chain.reset()
        assert all(marg.tensor is eval_.tensor for marg, eval_ in refs)
