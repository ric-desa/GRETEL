import torch, torch.nn as nn
from torch_geometric.nn.aggr import MeanAggregation,SoftmaxAggregation
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn import DenseGCNConv
from src.utils.torch.gcn import GCNConvWithEdgeTracking

from src.core.factory_base import build_w_params_string

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, t):
        # t: (batch,) int timesteps
        sinusoid = t[:, None] * self.inv_freq[None, :]
        emb = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)
        return emb  # (batch, dim)


class GCN_diffusion(nn.Module):
   
    def __init__(self, node_features, num_conv_layers=2, conv_booster=1, pooling=MeanAggregation, num_classes=2, time_dim=64):
        super(GCN_diffusion, self).__init__()

        self.time_dim = time_dim
        self.time_emb = TimeEmbedding(time_dim)
        self.num_classes = num_classes

        self.in_channels = node_features + num_classes + time_dim
        # print(f"GCN in_channels: {self.in_channels}")
        self.out_channels = int(self.in_channels * conv_booster)
          
        self.pooling =  build_w_params_string(pooling)
        
        # if num_conv_layers>1:
        #     self.num_conv_layers = [(self.in_channels, self.out_channels)] + [(self.out_channels, self.out_channels) * (num_conv_layers - 1)] # + [(self.out_channels, self.in_channels)]
        # else:
        #     self.num_conv_layers = [(self.in_channels, self.out_channels)] # + [(self.out_channels, self.in_channels)]
        # self.graph_convs = self.__init__conv_layers()

        if num_conv_layers > 1:
            self.num_conv_layers = [(self.in_channels, self.out_channels)] + \
                                [(self.out_channels, self.out_channels) for _ in range(num_conv_layers - 1)]
        else:
            self.num_conv_layers = [(self.in_channels, self.out_channels)]
        self.graph_convs = nn.ModuleList()
        for i, (in_ch, out_ch) in enumerate(self.num_conv_layers):
            # self.graph_convs.append(GCNConv(in_ch, out_ch, add_self_loops=True).float())
            self.graph_convs.append(GCNConvWithEdgeTracking(in_ch, out_ch, add_self_loops=True).float())
            # self.graph_convs.append(DenseGCNConv(in_ch, out_ch).float())
        
    def forward(self, node_features, edge_index, edge_weight, batch, y_t, t): # actually nodes_features here are node embeddings
        # print("- GCN super forward -")
        # print("node_features.shape:", node_features.shape)
        # print("y_t.shape:", y_t.shape)
        # print("t.shape:", t.shape)
        # print("batch.shape:", batch.shape)
        
        t_emb = self.time_emb(t).float()

        t_node = t_emb[batch]                         # [#nodes, time_dim]
        y_node = y_t[batch]                           # [#nodes, num_classes]

        # print("- Concat - ")
        # print("t_node.shape:", t_node.shape)
        # print("y_node.shape:", y_node.shape)

        node_features = torch.cat([node_features, y_node, t_node], dim=-1)  # [#nodes, ...]

        # print("After concat, node_features.shape:", node_features.shape)
        # print(f"initial node_features: {node_features}")
        # print(f"initial edge_index: {edge_index}")
        # print(f"initial edge_weight: {edge_weight}")

        edge_index = edge_index.long()
        for i, conv_layer in enumerate(self.graph_convs):
            node_features = conv_layer(node_features.float(), edge_index, edge_weight.float())
            # print(f"After conv layer {i}, node_features.shape:", node_features.shape)
            # print(f"node_features at {i}: {node_features}")
            node_features = nn.functional.relu(node_features)

        graph = node_features.clone()
        # print("Final graph.shape:", graph.shape)
    
        # global pooling
        if isinstance(self.graph_convs[-1],nn.Identity):
            return self.pooling(node_features)
        pool = self.pooling(node_features, batch)

        return graph, pool
    
    def __init__conv_layers(self):
        ############################################
        # initialize the convolutional layers interleaved with pooling layers
        graph_convs = []
        for i in range(len(self.num_conv_layers)):#add len
            graph_convs.append(GCNConv(in_channels=self.num_conv_layers[i][0],
                                      out_channels=self.num_conv_layers[i][1],
                                      add_self_loops=False).float())
        # graph_convs.append(self.pooling)
        return nn.Sequential(*graph_convs).float()

