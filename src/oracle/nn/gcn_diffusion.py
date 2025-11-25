import torch.nn as nn
from torch_geometric.nn.aggr import MeanAggregation

from src.utils.torch.gcn_diffusion import GCN_diffusion


class DownstreamGCN(GCN_diffusion):
   
    def __init__(self, node_features,
                 n_classes=2,
                 num_conv_layers=2,
                 num_dense_layers=2,
                 conv_booster=2,
                 linear_decay=2,
                 pooling=MeanAggregation()):
        
        super().__init__(node_features, num_conv_layers, conv_booster, pooling)
        
        self.num_dense_layers = num_dense_layers
        self.linear_decay = linear_decay
        self.n_classes = n_classes
        
        self.label_denoise = self.__init__downstream_layers(out=self.num_classes)
        self.graph_denoise = self.__init__downstream_layers(out=self.out_channels)
        self.adj_denoise = self.__init__downstream_layers(out=self.out_channels, adj=True)
        
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, node_features, edge_index, edge_weight, batch, y_t, t):
        graph, pool = super().forward(node_features, edge_index, edge_weight, batch, y_t, t)
        graph_pred = self.graph_denoise(graph)
        label_pred = self.label_denoise(pool)
        return graph_pred, label_pred
    
    def __init__downstream_layers(self, out, adj=False):
        ############################################
        # initialize the linear layers interleaved with activation functions
        downstream_layers = []
        in_linear = self.out_channels if not adj else self.out_channels*2 + self.time_dim
        for _ in range(self.num_dense_layers-1):
            downstream_layers.append(nn.Linear(in_linear, int(in_linear // self.linear_decay)))
            downstream_layers.append(nn.ReLU())
            in_linear = int(in_linear // self.linear_decay)
        # add the output layer
        downstream_layers.append(nn.Linear(in_linear, out))
        #downstream_layers.append(nn.Sigmoid())
        #downstream_layers.append(nn.Softmax())
        # put the linear layers in sequential
        return nn.Sequential(*downstream_layers).double()