import torch, torch.nn as nn
from torch_geometric.nn.aggr import MeanAggregation,SoftmaxAggregation
from torch_geometric.nn import GINEConv,MessagePassing
from torch_geometric.utils import softmax

from src.core.factory_base import build_w_params_string
from src.utils.norms import GNNLayerNormed

class GIN(nn.Module):
   
    def __init__(self, node_features, num_conv_layers=2, conv_booster=1, pooling=MeanAggregation, use_weights=True, class_="MultiHeadGGINConv"):
        super().__init__()
        
        self.in_channels = node_features
        self.out_channels = int(self.in_channels * conv_booster)          
        self.pooling =  build_w_params_string(pooling)
        
        if num_conv_layers>1:
            self.num_conv_layers = [(self.in_channels, self.out_channels)] + [(self.out_channels, self.out_channels)] * (num_conv_layers - 1)
        else:
            self.num_conv_layers = [(self.in_channels, self.out_channels)]
        
        self.graph_convs = self.__init__conv_layers(use_weights, class_)
        
    def forward(self, node_features, edge_index, edge_weights, batch):
        """
        node_features: [N_total, node_features]
        edge_index: [2, N_total^2] full adjacency indices
        edge_weights: [N_total^2] flattened adjacency (0 for missing edges)
        batch: [N_total] graph assignment vector
        """
        # convolution operations
        edge_index = edge_index.long()
        # Convert 1D edge_weights to 2D edge_attr
        edge_attr = edge_weights.unsqueeze(-1) # shape [num_edges, 1]

        for conv_layer in self.graph_convs[:-1]:
            # print(f"node_features.shape: {node_features.shape}")
            # print(f"edge_attr.shape: {edge_attr.shape}")
            node_features = conv_layer(node_features, edge_index, edge_attr, batch)
            node_features = nn.functional.relu(node_features)

        # global pooling
        if isinstance(self.graph_convs[-1],nn.Identity):
            return self.graph_convs[-1](node_features)

        return self.graph_convs[-1](node_features, batch)#, node_features
    
    def __init__conv_layers(self, use_weights=True, class_="MultiHeadGGINConv", edge_dim=1):
        layers = []
        for i in range(len(self.num_conv_layers)):

            mlp = nn.Sequential(
                nn.Linear(self.num_conv_layers[i][0], self.num_conv_layers[i][1]),
                nn.LeakyReLU(0.01),
                nn.Linear(self.num_conv_layers[i][1], self.num_conv_layers[i][1]),
                nn.LeakyReLU(0.01))

            if use_weights:
                # layers.append(GINEConvWithEdgeTracking(mlp, edge_dim=edge_dim).double())
                if class_ == "MultiHeadGGINConv":
                    # layers.append(MultiHeadGGINConv(mlp, edge_dim=1).double())
                    layers.append(GNNLayerNormed(MultiHeadGGINConv,
                                                 in_dim=mlp[0].in_features,
                                                 out_dim=mlp[0].out_features,
                                                 norm='graphnorm',
                                                 mlp=mlp,
                                                 edge_dim=1).double())
                elif class_ == "AGINConv":
                    # layers.append(AGINConv(mlp, num_heads=4, edge_dim=1).double())
                    layers.append(GNNLayerNormed(AGINConv,
                                                 in_dim=mlp[0].in_features,
                                                 out_dim=mlp[0].out_features,
                                                 norm='graphnorm',
                                                 mlp=mlp,
                                                 num_heads=4,
                                                 edge_dim=1).double())
                else: assert False, "Backbone class not recognized"
            else: layers.append(GINEConv(mlp, edge_dim=edge_dim).double())

        layers.append(self.pooling)
        return nn.Sequential(*layers).double()

class GINEConvWithEdgeTracking(GINEConv):
    def __init__(self, mlp, edge_dim=1):
        super().__init__(mlp, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr=None):
        # Ensure edge_attr has shape [num_edges, edge_dim]
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        # if edge_attr is not None:
        #     edge_attr = edge_attr.to(dtype=x.dtype, device=x.device)
        
        # print(f"x.shape: {x.shape}")
        # print(f"edge_attr.shape: {None if edge_attr is None else edge_attr.shape}")
        # print("mlp.in_features", self._modules[list(self._modules.keys())[0]].in_features)
        out = super().forward(x, edge_index, edge_attr)

        if edge_attr is not None:
            src = edge_index[0]
            self.last_messages = x[src] * edge_attr # Track messages for CF purposes

        return out

class AGINConv(MessagePassing):
    """
    Attention Graph Isomorphism Network convolution.
    Uses edge features for attention.
    """
    def __init__(self, in_features, out_features, mlp, num_heads=1, edge_dim=1, negative_slope=0.2):
        super().__init__(aggr="add")
        self.in_features = in_features # mlp[0].in_features
        self.out_features = out_features # mlp[0].out_features

        self.mlp = mlp
        self.edge_dim = edge_dim
        self.W = nn.Linear(self.in_features, self.in_features, bias=False)

        # Attention vector (single-head)
        att_dim = 2 * self.in_features + edge_dim
        # print(f"att_dim: {att_dim} and previous: {3 * self.in_features}")
        self.num_heads = num_heads
        self.att = nn.Parameter(torch.Tensor(num_heads, att_dim)) # multi-headed attention
        nn.init.xavier_uniform_(self.att.data)

        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(edge_dim, edge_dim)
        )

        self.residual = nn.Identity() if self.in_features == self.out_features else nn.Linear(self.in_features, self.out_features)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        # Pre-linear transform for attention projection
        x_proj = self.W(x)

        # Store for message()
        # self.x_proj = x_proj
        # self.edge_attr = edge_attr

        out = self.propagate(edge_index, x=x, x_proj=x_proj, edge_attr=edge_attr)
        return self.mlp(out) + self.residual(x)

    def message(self, x_j, x_i, x_proj_j, x_proj_i, edge_attr, index):
        edge_feat = self.edge_mlp(edge_attr)
        # Build attention input: [xi || xj || e_ij]
        att_input = torch.cat([x_proj_i, x_proj_j, edge_feat], dim=-1).unsqueeze(1) # [E,1,att_dim]

        # Raw attention score
        alpha = self.leaky_relu((att_input * self.att.unsqueeze(0)).sum(dim=-1)) # [E,num_heads]

        # Normalize over neighbors
        alpha = softmax(alpha, index) # softmax per edge per head

        # GIN-style message (but reweighted)
        msg = (x_j + edge_feat).unsqueeze(1) * alpha.unsqueeze(-1)  # [E,num_heads,node_feat_dim]
        msg = msg.mean(dim=1) # combine heads
        return msg
    
class GGINConv(MessagePassing):
    """
    Gated Graph Isomorphism Network convolution.
    Uses edge features for gating.
    """
    def __init__(self, mlp, edge_dim=1):
        super().__init__(aggr="add")  # GIN-style sum aggregation
        self.mlp = mlp
        self.edge_dim = edge_dim

        # Linear for edge gating
        self.edge_gate = nn.Linear(edge_dim, 1)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)  # [num_edges, edge_dim]

        return self.mlp(self.propagate(edge_index, x=x, edge_attr=edge_attr) + x)

    def message(self, x_j, edge_attr):
        """
        x_j: neighbor node features
        edge_attr: edge features for gating
        """
        # Compute gate per edge
        gate = torch.sigmoid(self.edge_gate(edge_attr))  # [num_edges, 1]

        # Weighted message
        msg = gate * x_j
        return msg
    
class MultiHeadGGINConv(MessagePassing):
    """
    Multi-head gated GIN convolution.
    Suitable for dense adjacency with many zero edges.
    """
    def __init__(self, in_dim, out_dim, mlp, edge_dim=1, num_heads=4):
        super().__init__(aggr="add")  # GIN-style sum
        self.mlp = mlp
        self.edge_dim = edge_dim
        self.num_heads = num_heads

        # One linear per head for edge gating
        self.edge_gates = nn.ModuleList([nn.Linear(edge_dim, 1) for _ in range(num_heads)])

        # Residual projection if needed
        # in_dim = mlp[0].in_features
        # out_dim = mlp[0].out_features
        self.residual = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)  # [num_edges, edge_dim]

        # Propagate per head and sum the heads
        head_msgs = []
        for head in range(self.num_heads):
            msg = self.propagate(edge_index, x=x, edge_attr=edge_attr, head=head)
            head_msgs.append(msg)
        out = torch.mean(torch.stack(head_msgs, dim=0), dim=0)  # average over heads

        # Residual connection
        out = self.mlp(out) + self.residual(x)
        return out

    def message(self, x_j, edge_attr, head):
        """
        Compute gated message for each edge and head.
        """
        gate = torch.sigmoid(self.edge_gates[head](edge_attr))  # [num_edges, 1]
        return gate * x_j


