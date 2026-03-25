import torch, torch.nn as nn
from torch_geometric.nn.aggr import MeanAggregation

from src.utils.torch.gcn_diffusion import GCN_diffusion


class DownstreamGCN_old(GCN_diffusion):
   
    def __init__(self, node_features,
                 n_classes=2,
                 num_conv_layers=2,
                 num_dense_layers=2,
                 conv_booster=2,
                 linear_decay=2,
                 pooling=MeanAggregation()):
        
        super().__init__(node_features, num_conv_layers, conv_booster, pooling, n_classes)
        
        self.num_node_features = node_features
        self.num_dense_layers = num_dense_layers
        self.linear_decay = linear_decay
        self.n_classes = n_classes
        
        self.label_denoise = self.__init__downstream_layers(out=self.n_classes)
        self.graph_denoise = self.__init__downstream_layers(out=self.num_node_features)
        self.adj_denoise = self.__init__downstream_layers(out=1, adj=True)
        
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
        edge_index = edge_index.long()
        graph, pool = super().forward(node_features, edge_index, edge_weight, batch, y_t, t)
        # print(f"graph, pool: {graph, pool}")
        graph_pred = self.graph_denoise(graph)
        # print(f"graph_pred.shape: {graph_pred.shape}")
        label_pred = self.label_denoise(pool)
        # print(f"label_pred.shape: {label_pred.shape}")

        src, dst = edge_index  # shape: [2, E]
        # print(f"edge_index.shape: {edge_index.shape}")
        edge_rep = torch.cat([graph[src], graph[dst]], dim=-1)
        # print(f"edge_rep.shape: {edge_rep.shape}")
        # shape: [E, 2*out_channels]
        weights_pred = self.adj_denoise(edge_rep).squeeze(-1)
        # shape: [E, 1]  (if your MLP outputs 1)

        # print(f"weights_pred.shape: {weights_pred.shape}")
        # print(f"graph_pred, label_pred, weights_pred: {graph_pred, label_pred, weights_pred}")
        return graph_pred, label_pred, weights_pred
    
    def __init__downstream_layers(self, out, adj=False):
        ############################################
        # initialize the linear layers interleaved with activation functions
        downstream_layers = []
        in_linear = self.out_channels if not adj else self.out_channels*2 # + self.time_dim
        for _ in range(self.num_dense_layers-1):
            downstream_layers.append(nn.Linear(in_linear, int(in_linear // self.linear_decay)))
            downstream_layers.append(nn.ReLU())
            in_linear = int(in_linear // self.linear_decay)
        # add the output layer
        downstream_layers.append(nn.Linear(in_linear, out))
        #downstream_layers.append(nn.Sigmoid())
        #downstream_layers.append(nn.Softmax())
        # put the linear layers in sequential
        return nn.Sequential(*downstream_layers).float()
    
class DownstreamGCN(GCN_diffusion):
    def __init__(self, node_features, n_classes=2, num_conv_layers=2, num_dense_layers=2, conv_booster=2, linear_decay=2, pooling=MeanAggregation()):
        super().__init__(node_features, num_conv_layers, conv_booster, pooling, n_classes)
        self.num_node_features = node_features
        self.num_dense_layers = num_dense_layers
        self.linear_decay = linear_decay
        self.n_classes = n_classes

        # downstream heads
        self.eps_label_head = self.__init__downstream_layers(out=self.n_classes, adj=False, graph_level=True)   # predicts noise for labels (eps)
        self.label_classifier = self.__init__downstream_layers(out=self.n_classes, adj=False, graph_level=True) # predicts logits for direct classification
        self.graph_denoise = self.__init__downstream_layers(out=self.num_node_features, adj=False, graph_level=False)
        self.adj_denoise = self.__init__downstream_layers(out=1, adj=True, graph_level=False)

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
        edge_index = edge_index.long()
        graph, pool = super().forward(node_features, edge_index, edge_weight, batch, t, y_t) if False else super().forward(node_features, edge_index, edge_weight, batch, y_t, t)
        # graph: [#nodes, channels], pool: [B, channels]

        # pool: [B, hidden_dim]
        B = batch.max().item() + 1

        # compute per-graph topology feature
        num_nodes = torch.bincount(batch, minlength=B)
        num_edges = edge_index.size(1) // 2  # undirected
        num_edges = torch.full((B,), num_edges // B, device=pool.device)

        cycle_rank = (num_edges - num_nodes + 1).float().unsqueeze(-1)

        pool = torch.cat([pool, cycle_rank], dim=-1)

        eps_X_pred = self.graph_denoise(graph)   # node-wise eps prediction
        # label eps predicted from pooled graph
        eps_y_pred = self.eps_label_head(pool)   # shape [B, n_classes]
        label_logits = self.label_classifier(pool)  # shape [B, n_classes]

        src, dst = edge_index
        edge_rep = torch.cat([graph[src], graph[dst]], dim=-1)
        eps_W_pred = self.adj_denoise(edge_rep).squeeze(-1)

        return eps_X_pred, eps_y_pred, eps_W_pred, label_logits
    
    def __init__downstream_layers(self, out, adj=False, graph_level=False):
        ############################################
        # initialize the linear layers interleaved with activation functions
        downstream_layers = []
        if graph_level:
            in_linear = self.out_channels + 1 # + cycle_rank
        elif adj:
            in_linear = self.out_channels * 2 # + self.time_dim
        else:
            in_linear = self.out_channels

        for _ in range(self.num_dense_layers-1):
            downstream_layers.append(nn.Linear(in_linear, int(in_linear // self.linear_decay)))
            downstream_layers.append(nn.ReLU())
            in_linear = int(in_linear // self.linear_decay)
        # add the output layer
        downstream_layers.append(nn.Linear(in_linear, out))
        #downstream_layers.append(nn.Sigmoid())
        #downstream_layers.append(nn.Softmax())
        # put the linear layers in sequential
        return nn.Sequential(*downstream_layers).float()
  