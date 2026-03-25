import torch, torch.nn as nn, torch_scatter
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, softmax

from src.core.factory_base import build_w_params_string
from src.utils.norms import GNNLayerNormed


class PNA(nn.Module):
   
    def __init__(self, node_features, num_conv_layers=2, conv_booster=1, pooling=MeanAggregation, use_weights=True):
        super(PNA, self).__init__()
        
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
                graph_convs.append(GNNLayerNormed(PNAConvWithEdgeTracking,
                                                  in_dim=self.num_conv_layers[i][0],
                                                  out_dim=self.num_conv_layers[i][1],
                                                  norm="graphnorm",
                                                  edge_dim=1).double())
            else:
                assert False, "PNA not implement for not using edge_weights!"
        graph_convs.append(self.pooling)
        return nn.Sequential(*graph_convs).double()

class PNAConvWithEdgeTracking(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=1,
                 aggregators=("mean", "max", "min", "std"),
                 scalers=("identity", "amplification", "attenuation"),
                 edge_mlp_hidden=16):

        super().__init__(aggr=None)  # we aggregate manually

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregators = aggregators
        self.scalers = scalers

        # Local message MLP
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_channels + edge_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

        # Combine all aggregators
        self.combine = nn.Linear(len(aggregators) * out_channels, out_channels)

        # Degree scalers
        self.eps = 1e-6

        # Edge refinement MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_mlp_hidden),
            nn.ReLU(),
            nn.Linear(edge_mlp_hidden, edge_dim)
        )

        # Residual projection
        self.residual = nn.Identity() if in_channels == out_channels else nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        row, col = edge_index
        deg = degree(row, num_nodes=x.size(0)).unsqueeze(-1)

        if edge_weight is not None:
            edge_weight = edge_weight.unsqueeze(-1)

        out = self.propagate(edge_index, x=x, deg=deg, edge_weight=edge_weight)
        out = self.combine(out)
        return out + self.residual(x)

    def message(self, x_j, edge_weight, deg):
        # refine edge features
        if edge_weight is None:
            ew = torch.zeros(x_j.size(0), 1, device=x_j.device)
        else:
            ew = self.edge_mlp(edge_weight) # (E, 1)

        msg = self.msg_mlp(torch.cat([x_j, ew], dim=-1)) # (E, F)\\
        return msg + ew # * (1 + ew) # * torch.sigmoid(ew)

    def aggregate(self, inputs, index, deg):
        aggs = []

        if "mean" in self.aggregators:
            aggs.append(torch_scatter.scatter(inputs, index, dim=0, reduce="mean"))
        if "max" in self.aggregators:
            aggs.append(torch_scatter.scatter(inputs, index, dim=0, reduce="max"))
        if "min" in self.aggregators:
            aggs.append(torch_scatter.scatter(inputs, index, dim=0, reduce="min"))
        if "std" in self.aggregators:
            mean = torch_scatter.scatter(inputs, index, dim=0, reduce="mean")
            var = torch_scatter.scatter((inputs - mean[index])**2, index, dim=0, reduce="mean")
            aggs.append(torch.sqrt(var + 1e-6))

        out = torch.cat(aggs, dim=-1)

        # Degree scalers
        if "amplification" in self.scalers:
            out = out * torch.log(deg + 1.0)
        if "attenuation" in self.scalers:
            out = out / torch.log(deg + 2.0)

        return out