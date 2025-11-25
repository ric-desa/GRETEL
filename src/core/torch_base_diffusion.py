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

        self.T = self.local_config['parameters']['T']
        self.undirected = self.local_config['parameters']['undirected']
        self.p_clean = self.local_config['parameters']['p_clean_diffusion_graph'] # diffusion probability to pass clean graph

        
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(self.device) 
        
        self.patience = 0                           
    
    def get_noise_schedule(self, T, beta_start=1e-4, beta_end=0.02):
        betas = torch.linspace(beta_start, beta_end, T, device=self.device)
        alphas = 1 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        return betas, alphas, alphas_bar

    def q_sample(self, x, t, alphas_bar):
        """
        x: [B, dim of input]
        t: [B] timesteps
        alphas_bar: [T]
        returns (y_t, noise)
        """
        sqrt_ab = alphas_bar[t]           # [B]
        sqrt_1_ab = torch.sqrt(1 - alphas_bar[t])
        noise = torch.randn_like(x)
        y_t = sqrt_ab[:,None] * x + sqrt_1_ab[:,None] * noise
        return y_t, noise

    @torch.no_grad()
    def sample_labels(self, model, graph_batch, T, alphas, alphas_bar, betas, y0):
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


    def real_fit(self):
              
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
                batch.batch = batch.batch.to(self.device)
                batch.x = node_features = batch.x.to(self.device)
                batch.edge_index = edge_index = batch.edge_index.to(self.device)
                batch.edge_attr = edge_weights = batch.edge_attr.to(self.device)
                print(node_features.shape, edge_weights.shape)
                # diffusion needs fully connected adjacency
                N = batch.x.shape[0]

                if self.undirected:
                    row = torch.arange(N, device=self.device).repeat_interleave(N)
                    col = torch.arange(N, device=self.device).repeat(N)
                    # mask_triu = col >= row
                    edge_index_full = torch.stack([row, col], dim=0)  # [2, E] or N(N-1) edges

                    W = torch.zeros((N, N), device=self.device)
                    src, dst = batch.edge_index
                    W[src, dst] = batch.edge_attr.float()
                    W[dst, src] = batch.edge_attr.float()
                    # row_ut, col_ut = row[mask_triu], col[mask_triu]
                    # edge_weights = W[row, col]  # W[row_ut, col_ut]     # [E] not triu
                    edge_weights = W.flatten() # NxN edges
                else:
                    ...

                labels = batch.y.to(self.device).long()
                print(f"training diffusion, # of classes: {self.dataset.num_classes}")
                y0 = torch.nn.functional.one_hot(labels, num_classes=self.dataset.num_classes).float()     # one-hot encode the labels
                
                self.optimizer.zero_grad()

                # diffusion
                t = torch.randint(0, self.T, (batch.num_graphs,), device=self.device) # TODO rivedere questo

                # q_sample to get noisy labels
                x_t, noise_X = self.q_sample(node_features, t, alphas_bar)
                w_t, noise_W = self.q_sample(edge_weights, t, alphas_bar)
                y_t, noise_y = self.q_sample(y0, t, alphas_bar)


                if torch.rand(1, device=self.device) < self.p_clean: # proability to see clean graph as "condition" | classifier-free gruidance
                    X_input = node_features
                    W_input = edge_weights
                else: 
                    X_input = x_t
                    W_input = w_t.flatten()


                # predict noise
                eps_X_pred, eps_y_pred, eps_W_pred = self.model(
                    node_features=X_input,
                    edge_index=edge_index_full,
                    edge_weight=W_input,
                    batch=batch.batch,
                    y_t=y_t,
                    t=t
                )
                # pred = self.model(node_features, edge_index, edge_weights, batch.batch)
                # print(f'pred.shape, labels.shape: {pred.shape, labels.shape}')
                               
                loss = self.loss_fn(eps_X_pred, noise_X) + self.loss_fn(eps_y_pred, noise_y) + self.loss_fn(eps_W_pred, noise_W)
                losses.append(loss.to('cpu').detach().numpy())
                loss.backward()

                #TODO
                pred, graph = torch.split(...)
                
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
                        y0 = torch.nn.functional.one_hot(labels, num_classes=self.dataset.num_classes).float()     # one-hot encode the labels

                        logits = self.sample_labels(self.model, batch, self.T, alphas, alphas_bar, betas, y0)

                        pred = logits.argmax(dim=-1)
                        
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
        local_config['parameters']['batch_size'] = local_config['parameters'].get('batch_size', 4)
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