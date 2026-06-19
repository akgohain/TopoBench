"""Unit tests for SGFormer."""

import pytest
import torch
import torch_geometric

from topobench.nn.backbones.graph.sgformer import (
    SGFormer,
    SGFormerAttention,
    SGFormerAttentionLayer,
    SGFormerGraphConv,
)
from topobench.nn.wrappers.graph import GNNWrapper


def test_sgformer_attention_layer_forward():
    """Test one simplified global attention layer."""
    x = torch.randn(6, 8)
    layer = SGFormerAttentionLayer(
        in_channels=8, out_channels=4, num_heads=2
    )

    out = layer(x, x)

    assert out.shape == (6, 4)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_sgformer_attention_returns_attentions():
    """Test dense attention inspection helper."""
    x = torch.randn(6, 8)
    model = SGFormerAttention(
        in_channels=8,
        hidden_channels=8,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
    )

    attentions = model.get_attentions(x)

    assert attentions.shape == (2, 6, 6)
    assert not torch.isnan(attentions).any()


def test_sgformer_graph_conv_forward(simple_graph_0):
    """Test the shallow graph convolution branch."""
    x = torch.randn(simple_graph_0.num_nodes, 8)
    model = SGFormerGraphConv(
        in_channels=8,
        hidden_channels=8,
        num_layers=2,
        dropout=0.0,
    )

    out = model(x, simple_graph_0.edge_index)

    assert out.shape == (simple_graph_0.num_nodes, 8)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_sgformer_forward_add(simple_graph_0):
    """Test SGFormer with additive branch aggregation."""
    x = torch.randn(simple_graph_0.num_nodes, 8)
    model = SGFormer(
        in_channels=8,
        hidden_channels=8,
        out_channels=8,
        trans_num_layers=1,
        gnn_num_layers=1,
        trans_dropout=0.0,
        gnn_dropout=0.0,
        aggregate="add",
    )

    out = model(x, simple_graph_0.edge_index)

    assert out.shape == (simple_graph_0.num_nodes, 8)
    assert not torch.isnan(out).any()


def test_sgformer_forward_cat(simple_graph_0):
    """Test SGFormer with concatenation branch aggregation."""
    x = torch.randn(simple_graph_0.num_nodes, 8)
    model = SGFormer(
        in_channels=8,
        hidden_channels=8,
        out_channels=4,
        trans_num_layers=1,
        gnn_num_layers=1,
        trans_dropout=0.0,
        gnn_dropout=0.0,
        aggregate="cat",
    )

    out = model(x, simple_graph_0.edge_index)

    assert out.shape == (simple_graph_0.num_nodes, 4)
    assert not torch.isnan(out).any()


def test_sgformer_without_graph_branch(simple_graph_0):
    """Test the global-attention-only SGFormer variant."""
    x = torch.randn(simple_graph_0.num_nodes, 8)
    model = SGFormer(
        in_channels=8,
        hidden_channels=8,
        out_channels=8,
        trans_dropout=0.0,
        gnn_dropout=0.0,
        use_graph=False,
    )

    out = model(x, simple_graph_0.edge_index)

    assert out.shape == (simple_graph_0.num_nodes, 8)


def test_sgformer_invalid_aggregate():
    """Test invalid aggregation mode validation."""
    with pytest.raises(ValueError, match="Invalid aggregate type"):
        SGFormer(in_channels=8, hidden_channels=8, aggregate="invalid")


def test_sgformer_gnn_wrapper(random_graph_input):
    """Test SGFormer through TopoBench's standard GNNWrapper."""
    x, _, _, edges_1, _ = random_graph_input
    batch = torch_geometric.data.Data(
        x_0=x,
        y=x,
        x=x,
        edge_index=edges_1,
        batch_0=torch.zeros(x.shape[0], dtype=torch.long),
    )
    model = SGFormer(
        in_channels=x.shape[1],
        hidden_channels=x.shape[1],
        out_channels=x.shape[1],
        trans_dropout=0.0,
        gnn_dropout=0.0,
    )
    wrapper = GNNWrapper(
        model,
        **{"out_channels": x.shape[1], "num_cell_dimensions": 1},
    )

    model_out = wrapper(batch)

    assert model_out["x_0"].shape == x.shape
    assert torch.equal(model_out["labels"], batch.y)
