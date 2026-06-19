"""SGFormer graph backbone.

This module implements the core architecture from SGFormer [1] for TopoBench.
SGFormer combines a simplified global-attention branch with a shallow graph
convolution branch, then aggregates the two node representations.

[1] Wu et al. "SGFormer: Simplifying and Empowering Transformers for
Large-Graph Representations." NeurIPS 2023.
Official implementation: https://github.com/qitianwu/SGFormer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree


class SGFormerGraphConvLayer(nn.Module):
    """SGFormer graph convolution layer with symmetric degree normalization.

    Parameters
    ----------
    in_channels : int
        Number of input features.
    out_channels : int
        Number of output features.
    use_weight : bool, optional
        Whether to apply a learnable linear transform after aggregation.
    use_init : bool, optional
        Whether to concatenate the initial representation before the linear
        transform, as in the reference implementation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_weight: bool = True,
        use_init: bool = False,
    ):
        super().__init__()
        self.use_init = use_init
        self.use_weight = use_weight
        linear_in_channels = 2 * in_channels if use_init else in_channels
        self.linear = nn.Linear(linear_in_channels, out_channels)

    def reset_parameters(self):
        """Reset learnable parameters."""
        self.linear.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``[num_nodes, in_channels]``.
        edge_index : torch.Tensor
            Graph connectivity in COO format of shape ``[2, num_edges]``.
        x0 : torch.Tensor
            Initial hidden representation used when ``use_init`` is enabled.

        Returns
        -------
        torch.Tensor
            Updated node features.
        """
        num_nodes = x.size(0)
        row, col = edge_index
        deg = degree(col, num_nodes, dtype=x.dtype).clamp_min(1)
        norm = deg[col].rsqrt() * deg[row].rsqrt()

        out = x.new_zeros(x.shape)
        out.index_add_(0, col, x[row] * norm.view(-1, 1))

        if self.use_init:
            out = torch.cat([out, x0], dim=-1)
            out = self.linear(out)
        elif self.use_weight:
            out = self.linear(out)

        return out


class SGFormerGraphConv(nn.Module):
    """Shallow graph convolution branch from SGFormer."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 1,
        dropout: float = 0.5,
        use_bn: bool = True,
        use_residual: bool = True,
        use_weight: bool = True,
        use_init: bool = False,
        use_act: bool = True,
    ):
        super().__init__()
        self.input_linear = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList(
            [
                SGFormerGraphConvLayer(
                    hidden_channels,
                    hidden_channels,
                    use_weight=use_weight,
                    use_init=use_init,
                )
                for _ in range(num_layers)
            ]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_channels) for _ in range(num_layers + 1)]
        )
        self.dropout = dropout
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        """Reset learnable parameters."""
        self.input_linear.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass."""
        x = self.input_linear(x)
        if self.use_bn:
            x = self.bns[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        initial_x = x
        previous_x = x
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, initial_x)
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x + previous_x
            previous_x = x
        return x


class SGFormerAttentionLayer(nn.Module):
    """Simplified linear-complexity global attention layer from SGFormer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int,
        use_weight: bool = True,
    ):
        super().__init__()
        self.key = nn.Linear(in_channels, out_channels * num_heads)
        self.query = nn.Linear(in_channels, out_channels * num_heads)
        self.value = (
            nn.Linear(in_channels, out_channels * num_heads)
            if use_weight
            else None
        )
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        """Reset learnable parameters."""
        self.key.reset_parameters()
        self.query.reset_parameters()
        if self.value is not None:
            self.value.reset_parameters()

    def forward(
        self,
        query_input: torch.Tensor,
        source_input: torch.Tensor,
        output_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        query = self.query(query_input).view(
            -1, self.num_heads, self.out_channels
        )
        key = self.key(source_input).view(
            -1, self.num_heads, self.out_channels
        )
        if self.use_weight:
            value = self.value(source_input).view(
                -1, self.num_heads, self.out_channels
            )
        else:
            value = source_input.view(-1, 1, self.out_channels)

        query = query / query.norm(p=2).clamp_min(1e-12)
        key = key / key.norm(p=2).clamp_min(1e-12)
        num_nodes = query.size(0)

        key_value = torch.einsum("nhm,nhd->hmd", key, value)
        attention_num = torch.einsum("nhm,hmd->nhd", query, key_value)
        attention_num = attention_num + num_nodes * value

        key_sum = key.sum(dim=0)
        attention_normalizer = torch.einsum("nhm,hm->nh", query, key_sum)
        attention_normalizer = attention_normalizer.unsqueeze(-1)
        attention_normalizer = attention_normalizer + num_nodes
        safe_normalizer = torch.where(
            attention_normalizer.abs() < 1e-12,
            torch.full_like(attention_normalizer, 1e-12),
            attention_normalizer,
        )
        attention_output = attention_num / safe_normalizer

        output = attention_output.mean(dim=1)
        if not output_attn:
            return output

        attention = torch.einsum("nhm,lhm->nlh", query, key).mean(dim=-1)
        normalizer = attention_normalizer.squeeze(-1).mean(
            dim=-1, keepdim=True
        )
        safe_normalizer = torch.where(
            normalizer.abs() < 1e-12,
            torch.full_like(normalizer, 1e-12),
            normalizer,
        )
        attention = attention / safe_normalizer
        return output, attention


class SGFormerAttention(nn.Module):
    """Global attention branch from SGFormer."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 1,
        num_heads: int = 1,
        dropout: float = 0.5,
        use_bn: bool = True,
        use_residual: bool = True,
        use_weight: bool = True,
        use_act: bool = True,
    ):
        super().__init__()
        self.input_linear = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList(
            [
                SGFormerAttentionLayer(
                    hidden_channels,
                    hidden_channels,
                    num_heads=num_heads,
                    use_weight=use_weight,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_channels) for _ in range(num_layers + 1)]
        )
        self.dropout = dropout
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        """Reset learnable parameters."""
        self.input_linear.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.input_linear(x)
        if self.use_bn:
            x = self.norms[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_outputs = [x]
        for i, conv in enumerate(self.convs):
            x = conv(x, x)
            if self.use_residual:
                x = (x + layer_outputs[i]) / 2.0
            if self.use_bn:
                x = self.norms[i + 1](x)
            if self.use_act:
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)
        return x

    def get_attentions(self, x: torch.Tensor) -> torch.Tensor:
        """Return dense attention matrices for inspection."""
        x = self.input_linear(x)
        if self.use_bn:
            x = self.norms[0](x)
        x = F.relu(x)

        layer_outputs = [x]
        attentions = []
        for i, conv in enumerate(self.convs):
            x, attention = conv(x, x, output_attn=True)
            attentions.append(attention)
            if self.use_residual:
                x = (x + layer_outputs[i]) / 2.0
            if self.use_bn:
                x = self.norms[i + 1](x)
            if self.use_act:
                x = F.relu(x)
            layer_outputs.append(x)
        return torch.stack(attentions, dim=0)


class SGFormer(nn.Module):
    """SGFormer backbone for graph-domain node representations.

    Parameters
    ----------
    in_channels : int
        Number of input node features.
    hidden_channels : int
        Hidden feature dimension.
    out_channels : int, optional
        Output feature dimension. If ``None``, defaults to ``hidden_channels``.
    trans_num_layers : int, optional
        Number of simplified global attention layers.
    trans_num_heads : int, optional
        Number of attention heads in the global branch.
    trans_dropout : float, optional
        Dropout rate in the global branch.
    trans_use_bn : bool, optional
        Whether to use layer normalization in the global branch.
    trans_use_residual : bool, optional
        Whether to use residual connections in the global branch.
    trans_use_weight : bool, optional
        Whether to learn value projections in global attention.
    trans_use_act : bool, optional
        Whether to apply ReLU after global attention layers.
    gnn_num_layers : int, optional
        Number of graph convolution layers.
    gnn_dropout : float, optional
        Dropout rate in the graph branch.
    gnn_use_weight : bool, optional
        Whether to use learnable graph-convolution weights.
    gnn_use_init : bool, optional
        Whether graph layers concatenate the initial hidden state.
    gnn_use_bn : bool, optional
        Whether to use batch normalization in the graph branch.
    gnn_use_residual : bool, optional
        Whether to use graph-branch residual connections.
    gnn_use_act : bool, optional
        Whether to apply ReLU after graph convolution layers.
    use_graph : bool, optional
        Whether to include the graph convolution branch.
    graph_weight : float, optional
        Weight of the graph branch when ``aggregate`` is ``"add"``.
    aggregate : str, optional
        Branch aggregation mode: ``"add"`` or ``"cat"``.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int | None = None,
        trans_num_layers: int = 1,
        trans_num_heads: int = 1,
        trans_dropout: float = 0.5,
        trans_use_bn: bool = True,
        trans_use_residual: bool = True,
        trans_use_weight: bool = True,
        trans_use_act: bool = True,
        gnn_num_layers: int = 1,
        gnn_dropout: float = 0.5,
        gnn_use_weight: bool = True,
        gnn_use_init: bool = False,
        gnn_use_bn: bool = True,
        gnn_use_residual: bool = True,
        gnn_use_act: bool = True,
        use_graph: bool = True,
        graph_weight: float = 0.8,
        aggregate: str = "add",
        **kwargs,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = hidden_channels

        self.trans_conv = SGFormerAttention(
            in_channels,
            hidden_channels,
            num_layers=trans_num_layers,
            num_heads=trans_num_heads,
            dropout=trans_dropout,
            use_bn=trans_use_bn,
            use_residual=trans_use_residual,
            use_weight=trans_use_weight,
            use_act=trans_use_act,
        )
        self.graph_conv = SGFormerGraphConv(
            in_channels,
            hidden_channels,
            num_layers=gnn_num_layers,
            dropout=gnn_dropout,
            use_bn=gnn_use_bn,
            use_residual=gnn_use_residual,
            use_weight=gnn_use_weight,
            use_init=gnn_use_init,
            use_act=gnn_use_act,
        )
        self.use_graph = use_graph
        self.graph_weight = graph_weight
        self.aggregate = aggregate
        self.out_channels = out_channels

        if aggregate == "add":
            projection_in_channels = hidden_channels
        elif aggregate == "cat":
            projection_in_channels = 2 * hidden_channels
        else:
            raise ValueError(f"Invalid aggregate type: {aggregate}")

        self.output_linear = nn.Linear(projection_in_channels, out_channels)

    def reset_parameters(self):
        """Reset learnable parameters."""
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.graph_conv.reset_parameters()
        self.output_linear.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass."""
        attention_x = self.trans_conv(x)
        if self.use_graph:
            graph_x = self.graph_conv(x, edge_index)
            if self.aggregate == "add":
                x = self.graph_weight * graph_x
                x = x + (1.0 - self.graph_weight) * attention_x
            else:
                x = torch.cat([attention_x, graph_x], dim=-1)
        else:
            x = attention_x
        return self.output_linear(x)

    def get_attentions(self, x: torch.Tensor) -> torch.Tensor:
        """Return global branch attention matrices for inspection."""
        return self.trans_conv.get_attentions(x)
