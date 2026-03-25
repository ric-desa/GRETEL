import torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn.aggr import MeanAggregation,SoftmaxAggregation,AttentionalAggregation
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn import GATConv

from src.core.factory_base import build_w_params_string
from src.utils.norms import GNNLayerNormed

from torch_geometric.utils import softmax, degree, add_self_loops


class GAT(nn.Module):
   
    def __init__(self, node_features, num_conv_layers=2, conv_booster=1, pooling=MeanAggregation, use_weights=True):
        super(GAT, self).__init__()
        
        self.in_channels = node_features        
        self.heads = 4 # Number of attention heads for GAT
        self.out_channels = int(self.in_channels * conv_booster)
          
        self.pooling =  build_w_params_string(pooling)
        # Attention gate
        self.att_gate = nn.Sequential(
            nn.Linear(self.out_channels*self.heads, self.out_channels),
            nn.ReLU(),
            nn.Linear(self.out_channels, self.out_channels // 2),
            nn.ReLU(),
            nn.Linear(self.out_channels // 2, 1)
        )
        for m in self.att_gate:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
        # Global Attention Pooling
        self.attention_pooling = AttentionalAggregation(self.att_gate).double()
 
        
        if num_conv_layers>1:
            self.num_conv_layers = [(self.in_channels, self.out_channels)] + [(self.out_channels*self.heads, self.out_channels) * (num_conv_layers - 1)]
        else:
            self.num_conv_layers = [(self.in_channels, self.out_channels)]
        self.graph_convs = self.__init__conv_layers(use_weights)
        
    def forward(self, node_features, edge_index, edge_weight, batch):
        # convolution operations
        edge_index = edge_index.long()
        # for conv_layer in self.graph_convs[:-1]:
        #     node_features = conv_layer(node_features, edge_index, edge_weight)
        #     node_features = nn.functional.relu(node_features)
        # node_features = nn.functional.dropout(node_features, p=0.1, training=self.training)
        for conv_layer in self.graph_convs[:-1]:
            if isinstance(conv_layer, GATConvMasked):
                edge_index, edge_weight = add_self_loops(edge_index, edge_attr=edge_weight)
                node_features = conv_layer(node_features, edge_index, edge_weight=edge_weight)
            elif isinstance(conv_layer, GATConv): node_features = conv_layer(node_features, edge_index)
            node_features = nn.functional.elu(node_features) # GAT usually works better with ELU
            node_features = nn.functional.dropout(node_features, p=0.1, training=self.training)
        # return node_features        

        # global pooling
        if isinstance(self.graph_convs[-1],nn.Identity):
            return self.graph_convs[-1](node_features)

        return node_features, self.graph_convs[-1](node_features, batch)
    
    def __init__conv_layers(self, use_weights):
        ############################################
        # initialize the convolutional layers interleaved with pooling layers
        graph_convs = []
        if not use_weights: 
            for i in range(len(self.num_conv_layers)):
                # graph_convs.append(GATConv(in_channels=self.num_conv_layers[i][0],
                #                         out_channels=self.num_conv_layers[i][1],
                #                         heads=self.heads,
                #                         concat=True,
                #                         dropout=0.1,
                #                         add_self_loops=True,
                #                         edge_dim=1
                #                         ).double())
                graph_convs.append(GNNLayerNormed(
                                GATConv,
                                in_dim=self.num_conv_layers[i][0],
                                out_dim=self.num_conv_layers[i][1],
                                norm="graphnorm",
                                heads=self.heads,
                                concat=True,
                                dropout=0.1,
                                add_self_loops=True,
                                edge_dim=1).double())
            graph_convs.append(self.pooling)

        else:
            for i in range(len(self.num_conv_layers)):
                graph_convs.append(GATConvMasked( # or GATConv
                                        in_channels=self.num_conv_layers[i][0],
                                        out_channels=self.num_conv_layers[i][1],
                                        heads=self.heads, # multiple attention heads
                                        concat=True,      # concatenates heads
                                        dropout=0.1,
                                        add_self_loops=False
                                        ).double())
            graph_convs.append(self.attention_pooling)

        return nn.Sequential(*graph_convs).double()
    
class GATConvMasked(GATConv):
    def __init__(self, in_channels, out_channels, heads=1, **kwargs):
        super().__init__(in_channels, out_channels, heads=heads, **kwargs)
        # self.residual = nn.Identity() if in_channels == out_channels * heads else nn.Linear(in_channels, out_channels * heads)

    def edge_update(self, alpha_j, alpha_i, edge_attr, index, ptr, dim_size,
                    edge_weight=None, deg_src=None):
        # Copy original GATConv logic:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        # Edge feature attention (optional)
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        # Original GAT nonlinearity
        alpha = F.leaky_relu(alpha, self.negative_slope)

        # MASKING: multiply attention logits by `edge_weight`
        if edge_weight is not None:
            # reshape so it broadcasts over heads
            edge_weight = edge_weight.view(-1, 1)
            alpha = alpha * edge_weight
            deg = deg_src[index].view(-1, 1) # [num_edges, 1]
            alpha /= (deg + 1e-6)

        # Continue original GAT
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return alpha

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None,
                size=None, return_attention_weights=None):

        # add self-loops
        # edge_index, edge_weight = add_self_loops(edge_index, edge_attr=edge_weight)

        # compute degree
        row, col = edge_index
        self.deg_src = degree(row, num_nodes=x.size(0), dtype=x.dtype)

        # Call parent forward but pass edge_weight to edge_update via kwargs
        self.edge_weight = edge_weight  # store temporarily
        out = super().forward(x, edge_index, edge_attr=edge_weight, size=size,return_attention_weights=return_attention_weights)
        self.edge_weight = None
        # print(f"out.shape: {out.shape} - residual(x).shape: {self.residual(x).shape}")
        return out # + self.residual(x)

    # Patch message passing to forward `edge_weight`
    def edge_updater(self, edge_index, **kwargs):
        return super().edge_updater(
            edge_index,
            **kwargs,
            edge_weight=self.edge_weight,
            deg_src=self.deg_src
        )