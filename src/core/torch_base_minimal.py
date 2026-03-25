import numpy as np
import random
import torch

from src.core.trainable_base import Trainable
from src.utils.cfg_utils import init_dflts_to_of
from src.core.factory_base import get_instance_kvargs
from sklearn.metrics import accuracy_score
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

class TorchBase(Trainable):
       
    def init(self):
        self.epochs = self.local_config['parameters']['epochs']
        self.batch_size = self.local_config['parameters']['batch_size']
        
        self.model = get_instance_kvargs(self.local_config['parameters']['model']['class'],
                                   self.local_config['parameters']['model']['parameters'])
        
        self.optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                      {'params':self.model.parameters(), **self.local_config['parameters']['optimizer']['parameters']})
        
        self.loss_fn = get_instance_kvargs(self.local_config['parameters']['loss_fn']['class'],
                                           self.local_config['parameters']['loss_fn']['parameters'])
        
        self.early_stopping_threshold = self.local_config['parameters']['early_stopping_threshold']
        
        self.lr_scheduler =  lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.5, total_iters=self.epochs)

        
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(self.device) 
        
        self.patience = 0                           
    
    def real_fit(self):
              
        instances = self.dataset.get_torch_instances(fold_id=self.fold_id)
        train_loader, val_loader = None, None
        
        if self.early_stopping_threshold:
            # num_instances = len(self.dataset.instances)
            num_instances = len(instances)
            # print(f'num_instances: {num_instances}')
            # get 5% of training instances and reserve them for validation
            indices = list(range(num_instances))
            random.shuffle(indices)
            val_size = max(int(.05 * len(indices)), self.batch_size)
            train_size = len(indices) - val_size
            # print(f'val_size, train_size: {val_size, train_size}')
            # get the training instances
            # train_instances = Subset(instances, indices[:train_size - 1])
            train_instances = Subset(instances, indices[:train_size])
            val_instances = Subset(instances, indices[train_size:])
            # get the train and validation loaders
            train_loader = DataLoader(train_instances, batch_size=self.batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_instances, batch_size=self.batch_size, shuffle=True, drop_last=True)
        else:
            train_loader = DataLoader(instances, batch_size=self.batch_size, shuffle=True, drop_last=True)

        best_loss = [0,0]
        
        for epoch in range(self.epochs):
            losses, preds, labels_list = [], [], []
            self.model.train()
            for batch in train_loader:
                # print("batch.shape:", batch.batch.shape)
                # print("unique graphs in batch:", batch.batch.unique())
                batch.batch = batch.batch.to(self.device)
                node_features = batch.x.to(self.device)
                edge_index = batch.edge_index.to(self.device)
                edge_weights = batch.edge_attr.to(self.device)
                edge_weights.requires_grad_(True)
                labels = batch.y.to(self.device).long()
                # print(f"labels at torch base: {labels[:5]}")
                # print("labels dtype:", labels.dtype, "min:", labels.min().item(), "max:", labels.max().item())
                
                self.optimizer.zero_grad()
                
                # pred, eps_W_pred = self.model(node_features, edge_index, edge_weights, batch.batch)
                pred = self.model(node_features, edge_index, edge_weights, batch.batch)
                # print(f'pred.shape, labels.shape: {pred.shape, labels.shape}')
                # print(f'pred: {pred[:5]}')
                if pred.shape[1] > 2: # and labels.dim() > 1:
                    labels = torch.nn.functional.one_hot(labels, num_classes=pred.shape[1]).float()                  
                    # print(f'labels after one hot: {labels[:5]}')
                loss_cls = self.loss_fn(pred, labels)


                edge_index_dropped = edge_index.clone()
                edge_weights_dropped = edge_weights.clone()

                # map each edge to a graph id
                edge_batch = batch.batch[edge_index[0]]  # [E]

                # for each graph, drop one random edge
                for g in edge_batch.unique():
                    idx = (edge_batch == g).nonzero(as_tuple=True)[0]
                    if idx.numel() > 0:
                        drop_e = idx[torch.randint(0, idx.numel(), (1,))]
                        edge_weights_dropped[drop_e] = 0.0   # soft drop (safer than removing)

                pred_dropped = self.model(node_features, edge_index_dropped, edge_weights_dropped, batch.batch)

                # TREE → CYCLE TARGETED FLIP
                edge_index_tc = edge_index.clone()
                edge_weights_tc = edge_weights.clone()

                for g in batch.batch.unique():
                    nodes = (batch.batch == g).nonzero(as_tuple=True)[0]

                    # only apply to trees
                    if nodes.numel() < 3:
                        continue

                    # pick two nodes already connected → adding edge creates a cycle
                    u, v = nodes[torch.randperm(nodes.numel())[:2]]

                    edge_index_tc = torch.cat([
                        edge_index_tc,
                        torch.tensor([[u, v], [v, u]], device=edge_index.device)
                    ], dim=1)

                    edge_weights_tc = torch.cat([
                        edge_weights_tc,
                        torch.ones(2, device=edge_weights.device)
                    ])
                
                pred_cycle = self.model(node_features, edge_index_tc, edge_weights_tc, batch.batch)

                def compute_cycle_mask_vectorized(edge_index, batch):
                    """
                    Ultra-fast vectorized per-graph cycle mask for batched graphs.
                    Marks edges that form cycles (1.0) or not (0.0).

                    Args:
                        edge_index: [2, E] long tensor
                        batch: Batch object with batch.batch [N] mapping nodes to graph ids

                    Returns:
                        cycle_mask: [E] float tensor, 1 if edge is on a cycle, 0 otherwise
                    """
                    device = edge_index.device
                    N = batch.batch.size(0)
                    E = edge_index.size(1)

                    # get per-edge graph ids
                    graph_ids = batch.batch[edge_index[0]]

                    # sort edges by graph to process per-graph in parallel
                    sorted_gids, perm = torch.sort(graph_ids)
                    sorted_edges = edge_index[:, perm]

                    # prepare output
                    cycle_mask = torch.zeros(E, device=device, dtype=torch.float)

                    # map global node indices to local per-graph indices
                    unique_graphs = sorted_gids.unique()
                    start = 0
                    for g in unique_graphs:
                        mask = (sorted_gids == g)
                        num_edges = mask.sum()
                        edges = sorted_edges[:, start : start + num_edges]
                        nodes, inverse = torch.unique(edges, return_inverse=True)
                        local_edges = inverse.view(2, -1)

                        # Union-Find (fast, tensor-based)
                        parent = torch.arange(nodes.size(0), device=device)

                        def find(u):
                            root = u
                            while True:
                                p = parent[root]
                                if p == root:
                                    break
                                root = p
                            while parent[u] != root:
                                next_u = parent[u]
                                parent[u] = root
                                u = next_u
                            return root

                        # mark cycles
                        for i in range(local_edges.size(1)):
                            u, v = local_edges[0, i].item(), local_edges[1, i].item()
                            pu, pv = find(u), find(v)
                            if pu == pv:
                                cycle_mask[perm[start + i]] = 1.0
                            else:
                                parent[pu] = pv
                        start += num_edges

                    return cycle_mask

                # CYCLE → TREE TARGETED FLIP
                cycle_mask = compute_cycle_mask_vectorized(edge_index, batch)

                edge_weights_ct = edge_weights.clone()
                edge_weights_ct[cycle_mask.bool()] = 0.0

                pred_tree = self.model(node_features, edge_index, edge_weights_ct, batch.batch)

                # Compute gradients of prediction w.r.t. edge weights
                # use integer labels (before one-hot conversion) for gradient target
                labels_int = batch.y.to(self.device).long()   # keep original label tensor

                # per-graph scalar: sum of the logit for the true class
                B = pred.size(0)
                scores = pred[torch.arange(B, device=pred.device), labels_int].sum()
                edge_grad = torch.autograd.grad(scores, edge_weights, create_graph=True)[0]

                edge_batch = batch.batch[edge_index[0]]
                assert edge_batch.shape[0] == edge_index.size(1)
                loss_grad_sparse = 0.0
                for g_id in batch.batch.unique():
                    mask = (edge_batch == g_id)        # you need edge_batch mapping
                    g_abs = edge_grad[mask].abs()
                    if g_abs.numel() > 2:
                        # Encourage only 1-2 edges to have high gradients
                        loss_grad_sparse += g_abs.sort(descending=True)[0][2:].sum()
                loss_grad_sparse /= batch.batch.unique().numel()

                # For cycle edges: encourage high gradients
                cycle_edges_grad = edge_grad[cycle_mask.bool()].abs()
                loss_cycle_grad = -cycle_edges_grad.mean() if cycle_edges_grad.numel() > 0 else 0.0

                # compute edge gradients
                # loss_cf = ((1 - cycle_mask) * eps_W_pred.abs()).mean()

                # g = eps_W_pred.abs()  # shape [E]
                # g_sorted, _ = g.sort(descending=True)
                # k = 5
                # loss_topk = g_sorted[k:].sum()

                # loss_flip = - (pred.softmax(-1) * pred_dropped.softmax(-1)).sum(dim=1).mean()

                # FLIP CONSISTENCY LOSS
                loss_flip_tc = -(pred.softmax(-1) * pred_cycle.softmax(-1)).sum(dim=1).mean()
                loss_flip_ct = -(pred.softmax(-1) * pred_tree.softmax(-1)).sum(dim=1).mean()

                λ_cf = 1e-2  # (1e-4 to 1e-2 range)
                λ_topk = 1e-4
                λ_flip = 1e-3
                λ_sparse = 1e-3 # encourage sparse gradients
                λ_cycle = 5e-4 # encourage high gradients on cycle edges
                loss = (
                    loss_cls 
                    # + λ_cf * loss_cf
                    # + λ_topk * loss_topk
                    + λ_sparse * loss_grad_sparse
                    + λ_cycle * loss_cycle_grad
                    + λ_flip * (loss_flip_tc + loss_flip_ct)
                    )
                
                # print("edge_grad.max()", edge_grad.abs().max().item())
                # print("loss_grad_sparse", loss_grad_sparse)
                # print("loss_cycle_grad", loss_cycle_grad)

                losses.append(loss.to('cpu').detach().numpy())
                loss.backward()
                
                labels_list += list(labels.squeeze().long().detach().to('cpu').numpy())
                preds += list(pred.squeeze().detach().to('cpu').numpy())
               
                self.optimizer.step()

            accuracy = self.accuracy(labels_list, preds)
            self.context.logger.info(f'epoch = {epoch} ---> loss = {np.mean(losses):.4f}\t accuracy = {accuracy:.4f}')
            self.lr_scheduler.step()
            
            # check if we need to do early stopping
            if self.early_stopping_threshold and len(val_loader) > 0:
                self.model.eval()
                var_losses, var_labels, var_preds = [], [], []
                with torch.no_grad():
                    for batch in val_loader:
                        batch.batch = batch.batch.to(self.device)
                        node_features = batch.x.to(self.device)
                        edge_index = batch.edge_index.to(self.device)
                        edge_weights = batch.edge_attr.to(self.device)
                        labels = batch.y.to(self.device).long()

                        pred = self.model(node_features, edge_index, edge_weights, batch.batch)
                        if pred.shape[1] > 2:
                            labels = torch.nn.functional.one_hot(labels, num_classes=pred.shape[1]).float()
                        loss = self.loss_fn(pred, labels)
                        
                        var_labels += list(labels.squeeze().to('cpu').numpy())
                        var_preds += list(pred.squeeze().to('cpu').numpy())
                        
                        var_losses.append(loss.item())
                        
                    best_loss.pop(0)
                    var_loss = np.mean(var_losses)
                    best_loss.append(var_loss)
                            
                    accuracy = self.accuracy(var_labels, var_preds)
                    self.context.logger.info(f'epoch = {epoch} ---> valid_loss = {var_loss:.4f}\t valid_accuracy = {accuracy:.4f}')
                
                if abs(best_loss[0] - best_loss[1]) < self.early_stopping_threshold:
                    self.patience += 1
                    
                    if self.patience == 4:
                        self.context.logger.info(f"Early stopped training at epoch {epoch}")
                        break  # terminate the training loop
                
    def check_configuration(self):
        super().check_configuration()
        local_config=self.local_config
        # set defaults
        local_config['parameters']['epochs'] = local_config['parameters'].get('epochs', 200)
        # local_config['parameters']['batch_size'] = local_config['parameters'].get('batch_size', 4)
        local_config['parameters']['batch_size'] = local_config['parameters'].get('batch_size', 1)
        local_config['parameters']['early_stopping_threshold'] = local_config['parameters'].get('early_stopping_threshold', None)
        # populate the optimizer
        init_dflts_to_of(local_config, 'optimizer', 'torch.optim.Adam',lr=0.001)
        init_dflts_to_of(local_config, 'loss_fn', 'torch.nn.BCELoss')
        
    def accuracy(self, testy, probs):
        # print(testy[:10], probs[:10])
        # testy_idx = np.argmax(testy, axis=1)
        # probs_idx = np.argmax(probs, axis=1)
        # print(testy_idx[:1], probs_idx[:1])
        # print(len(testy))

        if len(probs[0]) > 2:
            # print(np.argmax(testy, axis=1), np.argmax(testy, axis=1)-1, np.argmax(probs, axis=1))
            acc = accuracy_score(np.argmax(testy, axis=1), np.argmax(probs, axis=1))
        else:
            acc = accuracy_score(testy, np.argmax(probs, axis=1))
        return acc

    def read(self):
        super().read()
        if isinstance(self.model, list):
            for mod in self.model:
                mod.to(self.device)
        else:
            self.model.to(self.device)
            
    def to(self, device):
        if isinstance(self.model, torch.nn.Module):
            self.model.to(device)
        elif isinstance(self.model, list):
            for model in self.model:
                if isinstance(model, torch.nn.Module):
                    model.to(self.device)