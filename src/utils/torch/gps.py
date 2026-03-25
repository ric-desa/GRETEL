import torch.nn as nn
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GINConv
from torch_geometric.nn.conv import GCNConv
from torch.nn import MultiheadAttention

from src.core.factory_base import build_w_params_string
from src.utils.norms import GNNLayerNormed


class GPS(nn.Module):
   
    def __init__(self, node_features, num_conv_layers=2, conv_booster=1, pooling=MeanAggregation, use_weights=True):
        super(GPS, self).__init__()
        
        self.in_channels = node_features
        self.out_channels = int(self.in_channels * conv_booster)
          
        self.pooling =  build_w_params_string(pooling)
 
        if num_conv_layers>1:
            self.num_conv_layers = [(self.in_channels, self.out_channels)] + [(self.out_channels, self.out_channels) * (num_conv_layers - 1)]
        else:
            self.num_conv_layers = [(self.in_channels, self.out_channels)]
        self.graph_convs = self.__init__conv_layers(use_weights)
        
    def forward(self, node_features, edge_index, edge_weight, batch):
        # convolution operations
        edge_index = edge_index.long()
        for conv_layer in self.graph_convs[:-1]:
            node_features = conv_layer(node_features, edge_index, edge_weight, batch)
            node_features = nn.functional.relu(node_features)

        # global pooling
        if isinstance(self.graph_convs[-1],nn.Identity):
            return self.graph_convs[-1](node_features)

        return self.graph_convs[-1](node_features, batch)
    
    def __init__conv_layers(self, use_weights):
        ############################################
        # initialize the convolutional layers interleaved with pooling layers
        graph_convs = []
        for i in range(len(self.num_conv_layers)):#add len
            if use_weights:
                graph_convs.append(GNNLayerNormed(GPSLayer,
                                                  in_dim=self.num_conv_layers[i][0],
                                                  out_dim=self.num_conv_layers[i][1],
                                                  norm="graphnorm",
                                                  heads=1,
                                                  edge_dim=1).double())
            else:
                assert False, "GPS not implement for not using edge_weights!"
        graph_convs.append(self.pooling)
        return nn.Sequential(*graph_convs).double()

class GPSLayer(nn.Module):
    def __init__(self, dim_in, dim_out, heads=4, edge_dim=1,
                 local_gnn_type="gin"):
        super().__init__()

        # Local model (GIN recommended)
        mlp = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(),
            nn.Linear(dim_in, dim_in)
        )
        # self.local_gnn = GINConv(mlp)
        self.local_gnn = GCNConv(dim_in, dim_out)

        # Global transformer attention
        # print(f"dim_in: {dim_in}")
        self.global_attn = MultiheadAttention(
            embed_dim=dim_in,
            num_heads=heads,
            dropout=0.1,
            batch_first=True
        )

        # FFN block
        self.ffn = nn.Sequential(
            nn.Linear(dim_in, dim_in * 4),
            nn.ReLU(),
            nn.Linear(dim_in * 4, dim_in)
        )

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        self.norm3 = nn.LayerNorm(dim_in)

    def forward(self, x, edge_index, edge_weight, batch=None):
        # --- Local GNN ---
        x_local = self.local_gnn(x, edge_index, edge_weight)
        x = self.norm1(x + x_local)

        # --- Global transformer ---
        x_attn, _ = self.global_attn(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        x = self.norm2(x + x_attn.squeeze(0))

        # --- FFN ---
        x = self.norm3(x + self.ffn(x))

        return x
