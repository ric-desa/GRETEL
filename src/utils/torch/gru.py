import torch, torch.nn as nn, torch.nn.functional as F, torch_scatter
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from src.core.factory_base import build_w_params_string
from src.utils.norms import GNNLayerNormed


class GRU(nn.Module):
   
    def __init__(self, node_features, num_conv_layers=2, conv_booster=1, pooling=MeanAggregation, use_weights=True):
        super(GRU, self).__init__()
        
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
                graph_convs.append(GNNLayerNormed(GRUConvWithEdgeTracking,
                                                  in_dim=self.num_conv_layers[i][0],
                                                  out_dim=self.num_conv_layers[i][1],
                                                  norm="graphnorm",
                                                  edge_dim=1).double())
            else:
                assert False, "GRU-MPNN not implement for not using edge_weights!"
        graph_convs.append(self.pooling)
        return nn.Sequential(*graph_convs).double()

class GRUConvWithEdgeTracking(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=1, hidden_dim=None):
        super().__init__(aggr="add")  # we sum messages

        self.in_channels = in_channels
        self.out_channels = out_channels

        hidden_dim = out_channels if hidden_dim is None else hidden_dim

        # Message MLP φ(x_j, e_ij)
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_channels + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels)
        )

        # Edge gate ψ(e_ij)
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, 1),    # produces scalar gate per edge
            nn.Sigmoid()
        )

        # GRU update for node states
        self.gru = nn.GRUCell(out_channels, out_channels)

        # Residual if dimension mismatch
        self.residual = nn.Identity() if in_channels == out_channels else nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        # Prepare edge attributes
        if edge_weight is None:
            edge_weight = torch.zeros(edge_index.size(1), 1, device=x.device)
        else:
            edge_weight = edge_weight.unsqueeze(-1)

        # propagate() → message() → aggregate()
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        # GRU update (node-level)
        out = self.gru(out, self.residual(x))

        # track CF explanations (message magnitudes)
        self.last_messages = self._last_messages  # stored in message()

        return out

    def message(self, x_j, edge_weight):
        # edge refinement
        gate = self.edge_gate(edge_weight)           # (E,1)

        # store for CF
        self._last_messages = (x_j * gate)

        # message formulation
        msg_input = torch.cat([x_j, edge_weight], dim=-1)
        msg = self.msg_mlp(msg_input)                # (E,F)
        msg = msg * gate                             # gated messages

        return msg
