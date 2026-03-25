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

import math, os

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

        self.T = self.local_config['parameters']['T']
        self.undirected = self.local_config['parameters']['undirected']
        self.p_clean = self.local_config['parameters']['p_clean_diffusion_graph'] # diffusion probability to pass clean graph
        self.lambda_cls = self.local_config['parameters'].get('lambda_cls', 1.0) # balance diffusion and classification losses
        self.lambda_sparse = self.local_config['parameters'].get('lambda_sparse', 0.05) # balance loss on edge noise prediction

        
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(self.device) 
        
        self.patience = 0   

        # mixed precision scaler
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # early stopping bookkeeping
        self.best_val_loss = float('inf')
        self.best_state = None                   
    
    def get_noise_schedule(self, T, beta_start=1e-4, beta_end=0.02):
        betas = torch.linspace(beta_start, beta_end, T, device=self.device)
        alphas = 1 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        return betas, alphas, alphas_bar

    # def q_sample(self, x, t, alphas_bar):
    #     """
    #     x: [B, dim of input]
    #     t: [B] timesteps
    #     alphas_bar: [T]
    #     returns (y_t, noise)
    #     """
    #     sqrt_ab = alphas_bar[t]           # [B]
    #     sqrt_1_ab = torch.sqrt(1 - alphas_bar[t])
    #     noise = torch.randn_like(x)
    #     y_t = sqrt_ab[:,None] * x + sqrt_1_ab[:,None] * noise
    #     return y_t, noise

    # def q_sample(self, x, t_idx, alphas_bar, weights_flag=False):
    #     """
    #     x: [N, D] (N could be num_nodes or num_edges or batch size)
    #     t_idx: [N] int tensor mapping each row of x to a timestep
    #     alphas_bar: [T]
    #     returns (y_t, noise)
    #     """
    #     # alphas_bar[t_idx] -> (N,)
    #     sqrt_ab = torch.sqrt(alphas_bar[t_idx])          # (N,)
    #     sqrt_1_ab = torch.sqrt(1.0 - alphas_bar[t_idx])  # (N,)
    #     noise = torch.randn_like(x)
    #     if weights_flag:
    #         noise = abs(noise)
    #     y_t = sqrt_ab.unsqueeze(1) * x + sqrt_1_ab.unsqueeze(1) * noise
    #     return y_t, noise
    
    def q_sample(self, x, t_idx, alphas_bar, weights_flag=False):
        noise = torch.randn_like(x, device=self.device)
        if weights_flag:
            noise = noise.abs()
        sqrt_ab = torch.sqrt(alphas_bar[t_idx])
        sqrt_1_ab = torch.sqrt(1.0 - alphas_bar[t_idx])
        if x.dim() == 1:   # 1D edge weights
            y_t = sqrt_ab * x + sqrt_1_ab * noise
        else:              # node features
            y_t = sqrt_ab.unsqueeze(1) * x + sqrt_1_ab.unsqueeze(1) * noise
        return y_t, noise

    @torch.no_grad()
    def sample_labels_good(self, model, graph_batch, T, alphas, alphas_bar, betas, y0):
        B = graph_batch.num_graphs
        # start from pure Gaussian noise
        y_t = torch.randn(B, self.dataset.num_classes, device=self.device)
        for t in reversed(range(T)):
            t_vec = torch.full((B,), t, device=self.device, dtype=torch.long)
            # predict the noise
            eps_pred = model(
                x=graph_batch.x,
                edge_index=graph_batch.edge_index,
                edge_weight=graph_batch.edge_attr,
                batch=graph_batch.batch,
                y_t=y_t,
                t=t_vec
            )
            # DDPM update
            alpha_t = alphas[t]
            alpha_bar_t = alphas_bar[t]
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
            y_prev = coef1 * (y_t - coef2 * eps_pred)
            if t > 0:
                sigma_t = torch.sqrt(betas[t])
                y_prev += sigma_t * torch.randn_like(y0)
            y_t = y_prev
        # final discrete label
        logits = y_t  # or pass through a small MLP to get logits
        return logits#.argmax(dim=-1)

    @torch.no_grad()
    def sample_labels(self, model, graph_batch, T, alphas, alphas_bar, betas):
        """
        Batched DDPM sampling for label generation.
        Fully GPU-optimized, loop-free for fully connected graphs.
        """
        B = graph_batch.num_graphs
        N = graph_batch.x.size(0)

        device = self.device
        # start from pure Gaussian noise for labels
        y_t = torch.randn(B, self.dataset.num_classes, device=device)

        # Precompute fully-connected adjacency once for batch
        node_counts = torch.bincount(graph_batch.batch)
        cum_nodes = torch.cat([torch.tensor([0], device=device), node_counts.cumsum(0)])
        max_nodes = node_counts.max()
        arange_max = torch.arange(max_nodes, device=device)
        node_masks = arange_max.unsqueeze(0) < node_counts.unsqueeze(1)

        row_local = arange_max.repeat_interleave(max_nodes)
        col_local = arange_max.repeat(max_nodes)
        row_all = row_local.unsqueeze(0) + cum_nodes[:-1].unsqueeze(1)
        col_all = col_local.unsqueeze(0) + cum_nodes[:-1].unsqueeze(1)
        edge_mask = node_masks[:, row_local] & node_masks[:, col_local]
        row_list = row_all[edge_mask]
        col_list = col_all[edge_mask]
        edge_index_full = torch.stack([row_list, col_list], dim=0)
        edge_weights_full = torch.ones(edge_index_full.size(1), device=device)

        # Prepare per-node initial features
        node_features = graph_batch.x.to(device).float()
        batch_idx = graph_batch.batch.to(device)

        for t in reversed(range(T)):
            t_graph = torch.full((B,), t, device=device, dtype=torch.long)
            t_node = t_graph[batch_idx]
            t_edge = t_node[edge_index_full[0]]

            # Optional classifier-free guidance: p_clean
            if torch.rand(1, device=device) < self.p_clean:
                X_input = node_features
                W_input = edge_weights_full
            else:
                X_input = node_features  # or apply noise if desired
                W_input = edge_weights_full

            eps_X_pred, eps_y_pred, eps_W_pred = model(
                node_features=X_input,
                edge_index=edge_index_full,
                edge_weight=W_input,
                batch=batch_idx,
                y_t=y_t,
                t=t_node
            )

            # DDPM update
            alpha_t = alphas[t]
            alpha_bar_t = alphas_bar[t]
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
            y_prev = coef1 * (y_t - coef2 * eps_y_pred)

            if t > 0:
                sigma_t = torch.sqrt(betas[t])
                y_prev += sigma_t * torch.randn_like(y_prev)

            y_t = y_prev

        # Return final labels (logits)
        return y_t  # shape [B, num_classes]


    def real_fit_good(self):
              
        instances = self.dataset.get_torch_instances(fold_id=self.fold_id)
        train_loader, val_loader = None, None
        
        if self.early_stopping_threshold:
            num_instances = len(self.dataset.instances)
            # print(f'num_instances: {num_instances}')
            # get 5% of training instances and reserve them for validation
            indices = list(range(num_instances))
            random.shuffle(indices)
            val_size = max(int(.05 * len(indices)), self.batch_size)
            train_size = len(indices) - val_size
            # print(f'val_size, train_size: {val_size, train_size}')
            # get the training instances
            train_instances = Subset(instances, indices[:train_size - 1])
            val_instances = Subset(instances, indices[train_size:])
            # get the train and validation loaders
            train_loader = DataLoader(train_instances, batch_size=self.batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_instances, batch_size=self.batch_size, shuffle=True, drop_last=True)
        else:
            train_loader = DataLoader(instances, batch_size=self.batch_size, shuffle=True, drop_last=True)

        best_loss = [0,0]
        
        betas, alphas, alphas_bar = self.get_noise_schedule(self.T)
        for epoch in range(self.epochs):
            losses, preds, labels_list = [], [], []
            self.model.train()
            for batch in train_loader:
                # print(f"batch.shape: {batch.batch.shape}")
                # print(f"x.shape: {batch.x.shape}")
                # print(f"edge_index.shape: {batch.edge_index.shape}")
                # print(f"edge_attr.shape: {batch.edge_attr.shape}")
                batch.batch = batch.batch.to(self.device)
                batch.x = node_features = batch.x.to(self.device).float()
                batch.edge_index = edge_index = batch.edge_index.to(self.device)
                batch.edge_attr = edge_weights = batch.edge_attr.to(self.device).float()
                # print(f"node_features.shape, edge_weights.shape: {node_features.shape, edge_weights.shape}")
                # print(f"node_features, edge_weights: {node_features, edge_weights}")
                # diffusion needs fully connected adjacency
                N = batch.x.shape[0]

                # if self.undirected:
                #     row = torch.arange(N, device=self.device).repeat_interleave(N)
                #     col = torch.arange(N, device=self.device).repeat(N)
                #     # mask_triu = col >= row
                #     edge_index_full = torch.stack([row, col], dim=0)  # [2, E] or N(N-1) edges

                #     W = torch.zeros((N, N), device=self.device, dtype=torch.float)
                #     src, dst = batch.edge_index
                #     W[src, dst] = batch.edge_attr
                #     W[dst, src] = batch.edge_attr
                #     # row_ut, col_ut = row[mask_triu], col[mask_triu]
                #     # edge_weights = W[row, col]  # W[row_ut, col_ut]     # [E] not triu
                #     edge_weights = W.flatten().float() # NxN edges
                #     batch.edge_index = edge_index = edge_index_full
                #     # print(f"edge_weights.shape: {edge_weights.shape}")

                # else:
                #     ...

                labels = batch.y.to(self.device).long()
                # print(f"training diffusion, # of classes: {self.dataset.num_classes}")
                y0 = torch.nn.functional.one_hot(labels, num_classes=self.dataset.num_classes).float()     # one-hot encode the labels
                
                self.optimizer.zero_grad()

                # diffusion
                t = torch.randint(0, self.T, (batch.num_graphs,), device=self.device) # TODO review this
                
                t_node = t[batch.batch]   # shape: [num_nodes]
                # print(f"t_node.shape: {t_node.shape}")
                # print(f"edge_index.shape: {edge_index.shape}")

                if self.undirected:
                    edge_index_list = []
                    edge_weight_list = []
                    t_edge_list = []

                    start = 0
                    for g in range(batch.num_graphs):
                        node_mask = (batch.batch == g)
                        n_nodes = node_mask.sum().item()
                        
                        # Local node indices for this graph
                        local_nodes = torch.arange(n_nodes, device=self.device)
                        
                        # Fully connected edges within this graph
                        row = local_nodes.repeat_interleave(n_nodes) + start
                        col = local_nodes.repeat(n_nodes) + start
                        
                        edge_index_list.append(torch.stack([row, col], dim=0))
                        
                        # Edge weights: ones or your specific adjacency
                        edge_weight_list.append(torch.ones(row.shape[0], device=self.device, dtype=torch.float))
                        
                        # t_edge for each edge: take t_node of source nodes
                        t_edge_list.append(t_node[row])
                        
                        start += n_nodes

                    edge_index_full = torch.cat(edge_index_list, dim=1).to(self.device)  # [2, total_edges_in_batch]
                    edge_weights_full = torch.cat(edge_weight_list).to(self.device)      # [total_edges_in_batch]
                    t_edge_full = torch.cat(t_edge_list).to(self.device)                 # [total_edges_in_batch]
                    # print(f"edge_index_full.shape: {edge_index_full.shape}")
                    # print(f"edge_weights_full.shape: {edge_weights_full.shape}")
                    # print(f"t_edge_full.shape: {t_edge_full.shape}")                    

                # t_edge = t_node[edge_index_full[0]] # shape: [num_edges]
                # print(f"t_edge.shape: {t_edge.shape}")

                # q_sample to get noisy labels
                x_t, noise_X = self.q_sample(node_features, t_node, alphas_bar)
                # print(f"x_t.shape, noise_X.shape: {x_t.shape, noise_X.shape}")
                w_t, noise_W = self.q_sample(edge_weights_full, t_edge_full, alphas_bar, weights_flag=True)
                y_t, noise_y = self.q_sample(y0, t, alphas_bar)
                
                if torch.rand(1, device=self.device) < self.p_clean: # proability to see clean graph as "condition" | classifier-free gruidance
                    X_input = node_features
                    W_input = edge_weights_full
                    # TODO? y_input sometimes clean sometimes noised?
                else: 
                    X_input = x_t
                    W_input = w_t.flatten()

                # print(f"edge_index_full.shape: {edge_index_full.shape}")
                # print(f"X_input.shape: {X_input.shape}")
                # print(f"W_input.shape: {W_input.shape}")
                # predict noise
                # print(f"X_input, W_input, y_t: {X_input, W_input, y_t}")
                eps_X_pred, eps_y_pred, eps_W_pred = self.model(
                    node_features=X_input,
                    edge_index=edge_index_full,
                    edge_weight=W_input,
                    batch=batch.batch,
                    y_t=y_t,
                    t=t
                )
                # print(f"eps_X_pred.shape: {eps_X_pred.shape}")
                # print(f"eps_y_pred.shape: {eps_y_pred.shape}")
                # print(f"eps_W_pred.shape: {eps_W_pred.shape}")
                # pred = self.model(node_features, edge_index, edge_weights, batch.batch)
                # pred = eps_y_pred.argmax(dim=-1)
                # print(f"eps_y_pred: {eps_y_pred}")
                # print(f"pred: {pred}")
                # print(f'pred.shape, labels.shape: {pred.shape, labels.shape}')
                               
                loss = self.loss_fn(eps_X_pred, noise_X) + self.loss_fn(eps_y_pred, noise_y) + self.loss_fn(eps_W_pred, noise_W)
                losses.append(loss.to('cpu').detach().numpy())
                loss.backward()

                #TODO
                # pred, graph = torch.split(...)
                
                labels_list += labels.view(-1).long().detach().cpu().tolist()
                preds += eps_y_pred.detach().cpu().numpy().tolist()
               
                self.optimizer.step()
                # input("opt step completed")

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

                        B = batch.num_graphs
                        y_t = torch.randn(B, self.dataset.num_classes, device=self.device, dtype=torch.float)
                        for t in reversed(range(self.T)):
                            t_vec = torch.full((B,), t, device=self.device, dtype=torch.long)
                            eps_X_pred, eps_y_pred, eps_W_pred = self.model(
                                node_features = batch.x.to(self.device).float(),
                                edge_index = batch.edge_index.to(self.device),
                                edge_weight = batch.edge_attr.to(self.device).float(),
                                batch = batch.batch.to(self.device),
                                y_t = y_t,
                                t = t_vec
                            ) 
                            alpha_t = alphas[t]
                            alpha_bar_t = alphas_bar[t]
                            coef1 = 1.0 / torch.sqrt(alpha_t)
                            coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
                            y_prev = coef1 * (y_t - coef2 * eps_y_pred)
                            if t > 0:
                                sigma_t = torch.sqrt(betas[t])
                                y_prev = y_prev + sigma_t * torch.randn_like(y_prev, device=self.device, dtype=y_prev.dtype)
                            y_t = y_prev

                        logits = y_t
                        pred = logits.argmax(dim=-1).float()

                        if logits.shape[1] > 2:
                            labels = torch.nn.functional.one_hot(labels, num_classes=logits.shape[1]).float()          
                        loss = self.loss_fn(pred, labels)
                     
                        var_labels += list(labels.squeeze().to('cpu').numpy())
                        var_preds += list(logits.squeeze().to('cpu').numpy())
                        
                        var_losses.append(loss.to('cpu').detach().numpy())
                        
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


    def real_fit(self):
        instances = self.dataset.get_torch_instances(fold_id=self.fold_id)

        # train/val split (same logic but fewer surprises)
        if self.early_stopping_threshold:
            num_instances = len(self.dataset.instances)
            indices = list(range(num_instances))
            random.shuffle(indices)
            val_size = max(int(.05 * len(indices)), self.batch_size)
            train_size = len(indices) - val_size
            train_instances = Subset(instances, indices[:train_size])
            val_instances = Subset(instances, indices[train_size:])
            train_loader = DataLoader(train_instances, batch_size=self.batch_size, shuffle=True, drop_last=True, pin_memory=torch.cuda.is_available())
            val_loader = DataLoader(val_instances, batch_size=self.batch_size, shuffle=False, drop_last=False, pin_memory=torch.cuda.is_available())
        else:
            train_loader = DataLoader(instances, batch_size=self.batch_size, shuffle=True, drop_last=True, pin_memory=torch.cuda.is_available())
            val_loader = None

        # diffusion schedule
        betas, alphas, alphas_bar = self.get_noise_schedule(self.T)

        # classifier loss
        cls_loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            self.model.train()
            running_losses = []
            train_labels_all = []
            train_preds_all = []

            loop = train_loader
            for batch in loop:
                # move data
                batch.batch = batch.batch.to(self.device)
                node_features = batch.x.to(self.device).float()
                edge_index_raw = batch.edge_index.to(self.device)
                edge_attr_raw = batch.edge_attr.to(self.device).float()
                labels = batch.y.to(self.device).long()

                B = batch.num_graphs

                # sample t per graph

                t_graph = torch.randint(0, self.T, (B,), device=self.device)
                t_node = t_graph[batch.batch]

                if self.undirected:
                    edge_index_list = []
                    edge_weight_list = []
                    t_edge_list = []

                    start = 0
                    for g in range(batch.num_graphs):
                        node_mask = (batch.batch == g)
                        n_nodes = node_mask.sum().item()
                        
                        # Local node indices for this graph
                        local_nodes = torch.arange(n_nodes, device=self.device)
                        
                        # Fully connected edges within this graph
                        row = local_nodes.repeat_interleave(n_nodes) + start
                        col = local_nodes.repeat(n_nodes) + start
                        
                        edge_index_list.append(torch.stack([row, col], dim=0))
                        
                        # Edge weights: ones or your specific adjacency
                        edge_weight_list.append(torch.ones(row.shape[0], device=self.device, dtype=torch.float))
                        
                        # t_edge for each edge: take t_node of source nodes
                        t_edge_list.append(t_node[row])
                        
                        start += n_nodes

                    edge_index_full = torch.cat(edge_index_list, dim=1).to(self.device)  # [2, total_edges_in_batch]
                    edge_weights_full = torch.cat(edge_weight_list).to(self.device)      # [total_edges_in_batch]
                    t_edge_full = torch.cat(t_edge_list).to(self.device)                 # [total_edges_in_batch]

                # compute forward/noised versions
                x_t, noise_X = self.q_sample(node_features, t_node, alphas_bar)
                w_t, noise_W = self.q_sample(edge_weights_full, t_edge_full, alphas_bar, weights_flag=True)
                y0 = torch.nn.functional.one_hot(labels, num_classes=self.dataset.num_classes).float()
                y_t, noise_y = self.q_sample(y0, t_graph, alphas_bar)

                # classifier-free masking per-sample (per-graph)
                # create mask of shape [B] whether to keep clean or not
                keep_clean = (torch.rand(B, device=self.device) < self.p_clean)
                # broadcast to node-level (for X) if needed:
                # get per-node clean mask
                keep_clean_node = keep_clean[batch.batch]

                X_input = torch.where(keep_clean_node.unsqueeze(1), node_features, x_t)
                # repeats = torch.tensor([(batch.batch==g).sum().item() for g in range(B)], device=self.device, dtype=torch.long)
                # W_input = torch.where(keep_clean.repeat_interleave(repeats).unsqueeze(1),
                #                     edge_weights_full, w_t)
                node_counts = torch.tensor(
                    [(batch.batch==g).sum().item() for g in range(B)],
                    device=self.device
                )

                edge_counts = node_counts * node_counts   # fully-connected per graph

                keep_clean_edge = keep_clean.repeat_interleave(edge_counts)

                W_input = torch.where(
                    keep_clean_edge,
                    edge_weights_full,
                    w_t
                )

                Y_input = torch.where(keep_clean.unsqueeze(1), y0, y_t)

                self.optimizer.zero_grad()

                # forward with mixed precision
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    # model should now return (eps_X_pred, eps_y_pred, eps_W_pred, label_logits)
                    out = self.model(node_features=X_input, edge_index=edge_index_full, edge_weight=W_input, batch=batch.batch, y_t=Y_input, t=t_graph)
                    if isinstance(out, tuple) and len(out) == 4:
                        eps_X_pred, eps_y_pred, eps_W_pred, label_logits = out
                    else:
                        # fallback: (eps_X_pred, eps_y_pred, eps_W_pred)
                        eps_X_pred, eps_y_pred, eps_W_pred = out
                        label_logits = None

                    # diffusion losses (MSE)
                    λ = 0.1
                    mse_loss = λ * self.loss_fn(eps_X_pred, noise_X) + λ * self.loss_fn(eps_W_pred, noise_W) + λ * self.loss_fn(eps_y_pred, noise_y)
                    # classification loss
                    cls_loss = torch.nn.functional.cross_entropy(label_logits, labels)
                    
                    # Use one of the two sparsity losses:
                    # 1) sparsity loss on edge noise prediction
                    # sparse_loss = torch.mean(torch.abs(eps_W_pred))
                    # 2) edge-difference aware sparsity
                    delta_W = eps_W_pred * (edge_weights_full.abs() > 0).float()
                    sparse_loss = torch.mean(torch.abs(delta_W)) # edge_sparse_loss

                    # edge-count penalty
                    edge_l0 = torch.mean((eps_W_pred.abs() > 1e-3).float())

                    # total_loss = mse_loss + self.lambda_cls * cls_loss + self.lambda_sparse * sparse_loss
                    total_loss = mse_loss + self.lambda_cls * cls_loss + 0.05 * edge_l0

                    # classification loss (if label_logits provided)
                    if label_logits is not None:
                        cls_loss = cls_loss_fn(label_logits, labels)  # CrossEntropy on raw logits
                        total_loss = mse_loss + 0.5 * cls_loss  # weight CE as needed
                    else:
                        total_loss = mse_loss

                # backward + scaler
                self.scaler.scale(total_loss).backward()
                # gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                running_losses.append(total_loss.detach().cpu().item())

                # compute training accuracy using reconstructed y0_hat (from eps_y_pred) if available
                with torch.no_grad():
                    # reconstruct y0_hat per graph: shape [B, C]
                    sqrt_ab = torch.sqrt(alphas_bar[t_graph])[:, None]
                    sqrt_1_ab = torch.sqrt(1.0 - alphas_bar[t_graph])[:, None]
                    # y_t is current noised labels, eps_y_pred predicted noise
                    y0_hat = (Y_input - sqrt_1_ab * eps_y_pred) / (sqrt_ab + 1e-8)
                    pred_classes = y0_hat.argmax(dim=-1).cpu().numpy()
                    train_preds_all += pred_classes.tolist()
                    train_labels_all += labels.cpu().numpy().tolist()

            # epoch stats
            train_acc = accuracy_score(train_labels_all, train_preds_all) if len(train_preds_all) > 0 else 0.0
            avg_loss = float(np.mean(running_losses))
            self.context.logger.info(f"[Train] epoch {epoch}: loss {avg_loss:.4f}, acc {train_acc:.4f}")

            # validation
            if val_loader is not None:
                val_loss, val_acc = self._validate_epoch(val_loader, betas, alphas, alphas_bar, cls_loss_fn)
                self.context.logger.info(f"[Val] epoch {epoch}: val_loss {val_loss:.4f}, val_acc {val_acc:.4f}")

                # if not hasattr(self, "best_val_loss"):
                #     self.best_val_loss = -float("inf")

                # if val_loss > self.best_val_loss:
                #     self.best_val_loss = val_loss

                #     checkpoint = {
                #         "epoch": epoch,
                #         "model_state": self.model.state_dict(),
                #         "optimizer_state": self.optimizer.state_dict(),
                #         "scheduler_state": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                #         "val_acc": val_acc,
                #         "val_loss": val_loss,
                #         "config": self.local_config
                #     }

                # ckpt_path = os.path.join(self.context.get_output_dir(), "best_model.pt")
                # torch.save(checkpoint, ckpt_path)

                # self.context.logger.info(
                #     f"✓ Saved new best model checkpoint — accuracy improved to {val_acc:.4f}"
                # )

                # scheduler step (ReduceLROnPlateau wants metric)
                self.lr_scheduler.step()

                # early stopping
                if val_loss < self.best_val_loss - self.early_stopping_threshold:
                    self.best_val_loss = val_loss
                    self.best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                    self.patience = 0
                else:
                    self.patience += 1
                    if self.patience >= 10 and False:
                        self.context.logger.info(f"Early stopping at epoch {epoch}, best_val_loss {self.best_val_loss:.4f}")
                        break

    def real_fit_gpu_wrong(self):

        instances = self.dataset.get_torch_instances(fold_id=self.fold_id)

        # train/val split (same logic but fewer surprises)
        if self.early_stopping_threshold:
            num_instances = len(self.dataset.instances)
            indices = list(range(num_instances))
            random.shuffle(indices)
            val_size = max(int(.05 * len(indices)), self.batch_size)
            train_size = len(indices) - val_size
            train_instances = Subset(instances, indices[:train_size])
            val_instances = Subset(instances, indices[train_size:])
            train_loader = DataLoader(train_instances, batch_size=self.batch_size, shuffle=True, drop_last=True, pin_memory=torch.cuda.is_available())
            val_loader = DataLoader(val_instances, batch_size=self.batch_size, shuffle=False, drop_last=False, pin_memory=torch.cuda.is_available())
        else:
            train_loader = DataLoader(instances, batch_size=self.batch_size, shuffle=True, drop_last=True, pin_memory=torch.cuda.is_available())
            val_loader = None

        # diffusion schedule
        betas, alphas, alphas_bar = self.get_noise_schedule(self.T)

        # classifier loss
        cls_loss_fn = torch.nn.CrossEntropyLoss()

        scaler = torch.cuda.amp.GradScaler(enabled=True)
        cls_loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(self.epochs):

            for batch in train_loader:

                # -------------------------------
                # Move tensors to GPU
                # -------------------------------
                batch.batch = batch.batch.to(self.device)
                node_features = batch.x.to(self.device).float()
                edge_index_raw = batch.edge_index.to(self.device)
                edge_attr_raw = batch.edge_attr.to(self.device).float()
                labels = batch.y.to(self.device).long()

                B = batch.num_graphs   # number of graphs in batch
                N = node_features.size(0)

                # ---------------------------------------------------
                # 1) Sample diffusion time for each graph
                # ---------------------------------------------------
                t_graph = torch.randint(0, self.T, (B,), device=self.device)
                t_node = t_graph[batch.batch]

                # ---------------------------------------------------
                # 2) Fully-connected general adjacency (GPU-only)
                # ---------------------------------------------------
                if self.undirected:

                    # Per-graph node counts (variable size)
                    node_counts = torch.bincount(batch.batch)
                    max_nodes = node_counts.max()

                    # [max_nodes]
                    arange_max = torch.arange(max_nodes, device=self.device)

                    # Mask: which nodes exist in each graph
                    node_masks = (arange_max.unsqueeze(0) <
                                node_counts.unsqueeze(1))  # [B, max_nodes]

                    # Local (i,j) pairs for full FC adjacency
                    row_local = arange_max.repeat_interleave(max_nodes)   # [max_nodes^2]
                    col_local = arange_max.repeat(max_nodes)              # [max_nodes^2]

                    # Graph offsets: cumulative node starts
                    cum_nodes = torch.cat([
                        torch.tensor([0], device=self.device),
                        node_counts.cumsum(0)
                    ])  # [B+1]

                    # Broadcast local pairs into graph-global index space
                    # Shapes: [B, max_nodes^2]
                    row_all = row_local.unsqueeze(0) + cum_nodes[:-1].unsqueeze(1)
                    col_all = col_local.unsqueeze(0) + cum_nodes[:-1].unsqueeze(1)

                    # Mask out invalid edges
                    edge_mask = node_masks[:, row_local] & node_masks[:, col_local]

                    # Final batched edges (flatten)
                    row_list = row_all[edge_mask]
                    col_list = col_all[edge_mask]
                    edge_index_full = torch.stack([row_list, col_list], dim=0)

                    # Constant edge weights or custom
                    edge_weights_full = torch.ones(edge_index_full.size(1),
                                                device=self.device)

                    # Per-edge diffusion time (source node)
                    src_nodes = edge_index_full[0]
                    t_edge_full = t_node[src_nodes]

                else:
                    # Use original sparse edges
                    edge_index_full = edge_index_raw
                    edge_weights_full = edge_attr_raw.squeeze()
                    src_nodes = edge_index_full[0]
                    t_edge_full = t_node[src_nodes]

                # ---------------------------------------------------
                # 3) Diffusion q_sample for X, W, Y
                # ---------------------------------------------------
                x_t, noise_X = self.q_sample(node_features, t_node, alphas_bar)

                w_t, noise_W = self.q_sample(edge_weights_full,
                                            t_edge_full,
                                            alphas_bar,
                                            weights_flag=True)

                y0 = torch.nn.functional.one_hot(
                    labels, num_classes=self.dataset.num_classes
                ).float()

                y_t, noise_y = self.q_sample(y0, t_graph, alphas_bar)

                # ---------------------------------------------------
                # 4) Clean/noisy masks
                # ---------------------------------------------------
                # per-graph clean flags
                keep_clean = (torch.rand(B, device=self.device) < self.p_clean)

                # per-node clean mask
                keep_clean_node = keep_clean[batch.batch]     # [N]

                # per-edge clean mask
                if self.undirected:
                    edge_counts = node_counts * node_counts
                    keep_clean_edge = keep_clean.repeat_interleave(edge_counts)
                else:
                    keep_clean_edge = keep_clean.repeat_interleave(
                        torch.bincount(src_nodes)
                    )

                # ---------------------------------------------------
                # 5) Select input to diffusion model (clean or noisy)
                # ---------------------------------------------------
                X_input = torch.where(keep_clean_node.unsqueeze(1),
                                    node_features, x_t)

                W_input = torch.where(keep_clean_edge,
                                    edge_weights_full, w_t)

                Y_input = torch.where(keep_clean.unsqueeze(1),
                                    y0, y_t)   # one-hot targets

                # ---------------------------------------------------
                # 6) Forward model
                # ---------------------------------------------------
                self.optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=True):
                    pred_X, pred_W, pred_y, label_logits = self.model(
                        X_input,
                        edge_index_full,
                        W_input,
                        batch.batch,
                        Y_input,
                        t_node
                    )

                    # diffusion losses
                    loss_x = torch.nn.functional.mse_loss(pred_X, noise_X)
                    noise_W = noise_W.flatten()
                    pred_W = pred_W.flatten()
                    loss_w = torch.nn.functional.mse_loss(pred_W, noise_W)
                    loss_y = torch.nn.functional.mse_loss(pred_y, noise_y)

                    diffusion_loss = (
                        self.loss_weight_x * loss_x +
                        self.loss_weight_w * loss_w +
                        self.loss_weight_y * loss_y
                    )

                    # classification loss (raw logits vs class IDs)
                    cls_loss = cls_loss_fn(label_logits, labels)

                    loss = diffusion_loss + self.lambda_cls * cls_loss

                # ---------------------------------------------------
                # 7) backward + gradient clipping + optimizer step
                # ---------------------------------------------------
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(self.optimizer)
                scaler.update()

            # End epoch
            print(f"[Epoch {epoch}] Loss={loss.item():.4f} "
                f"Diff={diffusion_loss.item():.4f} Cls={cls_loss.item():.4f}")


    @torch.no_grad()
    def _validate_epoch(self, val_loader, betas, alphas, alphas_bar, cls_loss_fn):
        self.model.eval()
        val_losses = []
        all_preds = []
        all_labels = []
        for batch in val_loader:
            batch.batch = batch.batch.to(self.device)
            node_features = batch.x.to(self.device).float()
            edge_index = batch.edge_index.to(self.device)
            edge_weights = batch.edge_attr.to(self.device).float()
            labels = batch.y.to(self.device).long()
            B = batch.num_graphs

            # sample starting y_t randomly and run reverse denoising chain (single determinisitc step using model's eps)
            y_t = torch.randn(B, self.dataset.num_classes, device=self.device)
            for t in reversed(range(self.T)):
                t_vec = torch.full((B,), t, device=self.device, dtype=torch.long)
                out = self.model(node_features=node_features, edge_index=edge_index, edge_weight=edge_weights, batch=batch.batch, y_t=y_t, t=t_vec)
                if isinstance(out, tuple) and len(out) == 4:
                    eps_X_pred, eps_y_pred, eps_W_pred, label_logits = out
                else:
                    eps_X_pred, eps_y_pred, eps_W_pred = out
                    label_logits = None

                # DDPM single-step update
                alpha_t = alphas[t]
                alpha_bar_t = alphas_bar[t]
                coef1 = 1.0 / math.sqrt(alpha_t)
                coef2 = (1.0 - alpha_t) / math.sqrt(1.0 - alpha_bar_t)
                y_prev = coef1 * (y_t - coef2 * eps_y_pred)
                if t > 0:
                    sigma_t = math.sqrt(betas[t])
                    y_prev = y_prev + sigma_t * torch.randn_like(y_prev)
                y_t = y_prev

            # compute losses: reconstruct y0_hat from last eps
            # if label_logits exists use crossentropy for val loss else MSE on reconstruction vs one-hot
            if label_logits is not None:
                cls_loss = cls_loss_fn(label_logits, labels)
                val_losses.append(cls_loss.item())
                preds = label_logits.argmax(dim=-1).cpu().numpy()
            else:
                # reconstruct from eps_y_pred at t=0 logic (approx)
                # here we use the final y_t logits from chain
                preds = y_t.argmax(dim=-1).cpu().numpy()
                val_losses.append(0.0)

            all_preds += preds.tolist()
            all_labels += labels.cpu().numpy().tolist()

        val_acc = accuracy_score(all_labels, all_preds) if len(all_preds) else 0.0
        val_loss = float(np.mean(val_losses)) if len(val_losses) else 0.0
        return val_loss, val_acc


    def check_configuration(self):
        super().check_configuration()
        local_config=self.local_config
        # set defaults
        local_config['parameters']['epochs'] = local_config['parameters'].get('epochs', 200)
        local_config['parameters']['batch_size'] = local_config['parameters'].get('batch_size', 4)
        local_config['parameters']['early_stopping_threshold'] = local_config['parameters'].get('early_stopping_threshold', None)
        # populate the optimizer
        init_dflts_to_of(local_config, 'optimizer', 'torch.optim.Adam',lr=0.001)
        init_dflts_to_of(local_config, 'loss_fn', 'torch.nn.BCELoss')
        
    def accuracy_good(self, testy, probs):
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
    
    def accuracy(self, testy, probs):
        # expects testy as array of labels, probs as logits or probabilities
        if isinstance(probs, np.ndarray) and probs.ndim > 1:
            return accuracy_score(testy, np.argmax(probs, axis=1))
        else:
            return accuracy_score(testy, np.array(probs).astype(int))


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