import numpy as np
import torch

from src.core.oracle_base import Oracle
from src.core.torch_base_diffusion import TorchBase
from src.dataset.utils.dataset_torch import TorchGeometricDataset

import math
from sklearn.metrics import accuracy_score

class OracleTorch(TorchBase, Oracle):     
            
    def real_fit(self):
        super().real_fit()
        self.evaluate(self.dataset, fold_id=self.fold_id)

    # def get_noise_schedule(self, T, beta_start=1e-4, beta_end=0.02):
    #     betas = torch.linspace(beta_start, beta_end, T, device=self.device)
    #     alphas = 1 - betas
    #     alphas_bar = torch.cumprod(alphas, dim=0)
    #     return betas, alphas, alphas_bar
    
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
            
    @torch.no_grad()
    def evaluate_good(self, dataset, fold_id=0):            
        loader = dataset.get_torch_loader(fold_id=fold_id, batch_size=self.batch_size, usage='test')
        
        losses = []
        labels_list, preds = [], []
        self.T = self.local_config['parameters']['T'] 
        betas, alphas, alphas_bar = TorchBase.get_noise_schedule(self, self.T)
        for batch in loader:
            batch.batch = batch.batch.to(self.device)
            node_features = batch.x.to(self.device)
            edge_index = batch.edge_index.to(self.device)
            edge_weights = batch.edge_attr.to(self.device)
            labels = batch.y.to(self.device).long()

            # y0 = torch.nn.functional.one_hot(labels, num_classes=self.dataset.num_classes).float()     # one-hot encode the labels
            # t = torch.randint(0, self.T, (batch.num_graphs,), device=self.device)
            # y_t, noise_y = TorchBase.q_sample(self, y0, t, alphas_bar)

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
            # print(f"pred.shape: {pred.shape}")

            
            self.optimizer.zero_grad()  
            # eps_X_pred, eps_y_pred, eps_W_pred = self.model(node_features, edge_index, edge_weights, batch.batch, y_t, t)
            if logits.shape[1] > 2:
                labels = torch.nn.functional.one_hot(labels, num_classes=logits.shape[1]).float()          
            loss = self.loss_fn(pred, labels)
            losses.append(loss.to('cpu').detach().numpy())
            
            labels_list += labels.view(-1).long().detach().cpu().tolist()
            preds += logits.detach().cpu().numpy().tolist()
            
        accuracy = self.accuracy(labels_list, preds)
        self.context.logger.info(f'Test accuracy = {np.mean(accuracy):.4f}')

    @torch.no_grad()
    def evaluate(self, dataset, fold_id=0):
        val_loader = dataset.get_torch_loader(fold_id=fold_id, batch_size=self.batch_size, usage='test')

        betas, alphas, alphas_bar = TorchBase.get_noise_schedule(self, self.T)

        cls_loss_fn = torch.nn.CrossEntropyLoss()

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
            # y_t = torch.randn(B, self.dataset.num_classes, device=self.device)
            y_t = torch.zeros(B, self.dataset.num_classes, device=self.device) # deterministic
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
                    # sigma_t = math.sqrt(betas[t]) # stochastic
                    # y_prev = y_prev + sigma_t * torch.randn_like(y_prev) # stochastic
                    pass
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

    def _real_predict(self, data_instance):
        return torch.argmax(self._real_predict_gradients_diffusion(data_instance), dim=-1)
    
    def _real_predict_gradients(self, data_instance):
        '''We added this since we needed gradients to compute how the perturbation on P influences Loss'''
        return self._real_predict_proba_gradients_diffusion(data_instance)

    @torch.no_grad()
    def _real_predict_proba(self, data_inst):
        # print(f"type torch.py ln46: {type(data_inst.data)}")
        # instance_data_np = np.array(data_inst.data) if isinstance(data_inst.data, memoryview) else data_inst.data
        data_inst = TorchGeometricDataset.to_geometric(data_inst)
        node_features = data_inst.x.to(self.device)
        edge_index = data_inst.edge_index.to(self.device)
        edge_weights = data_inst.edge_attr.to(self.device)
        
        return self.model(node_features,edge_index,edge_weights, None).cpu().squeeze()
    
    # @torch.no_grad() # We need gradients even at inference
    def _real_predict_proba_gradients(self, data_inst):
        # print(f"type torch.py ln46: {type(data_inst.data)}")
        # instance_data_np = np.array(data_inst.data) if isinstance(data_inst.data, memoryview) else data_inst.data
        data_inst = TorchGeometricDataset.to_geometric_gradients(data_inst)
        node_features = data_inst.x.to(self.device)
        edge_index = data_inst.edge_index.to(self.device)
        edge_weights = data_inst.edge_attr.to(self.device)
        
        return self.model(node_features,edge_index,edge_weights, None).cpu().squeeze()
    
    def _real_predict_gradients_diffusion(self, data_inst):
        """Compute prediction using the diffusion oracle for a single instance, keeping gradients."""
        data_inst = TorchGeometricDataset.to_geometric_gradients(data_inst)
        
        node_features = data_inst.x.to(self.device).float()
        edge_index = data_inst.edge_index.to(self.device)
        edge_weights = data_inst.edge_attr.to(self.device).float()
        
        # Single graph -> batch tensor of zeros
        batch = torch.zeros(node_features.shape[0], dtype=torch.long, device=self.device)

        # Initialize noisy labels
        B = 1
        y_t = torch.randn(B, self.dataset.num_classes, device=self.device, dtype=torch.float)

        betas, alphas, alphas_bar = self.get_noise_schedule(self.T)

        # Reverse diffusion loop
        for t in reversed(range(self.T)):
            t_vec = torch.full((B,), t, device=self.device, dtype=torch.long)
            out = self.model(
                node_features=node_features,
                edge_index=edge_index,
                edge_weight=edge_weights,
                batch=batch,
                y_t=y_t,
                t=t_vec
            )
            if isinstance(out, tuple) and len(out) == 4:
                eps_X_pred, eps_y_pred, eps_W_pred, label_logits = out
            else:
                eps_X_pred, eps_y_pred, eps_W_pred = out
                label_logits = None
            alpha_t = alphas[t]
            alpha_bar_t = alphas_bar[t]
            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
            y_prev = coef1 * (y_t - coef2 * eps_y_pred)
            if t > 0:
                sigma_t = torch.sqrt(betas[t])
                y_prev = y_prev + sigma_t * torch.randn_like(y_prev, device=self.device, dtype=y_prev.dtype)
            y_t = y_prev

        # Return logits (keep gradients for explainer)
        return y_t.squeeze()
                     
    def check_configuration(self):#TODO: revise configuration
        super().check_configuration()
        local_config = self.local_config

        if 'model' not in local_config['parameters']:
            local_config['parameters']['model'] = {
                'class': "src.oracle.nn.gcn.DownstreamGCN",
                "parameters" : {}
            }

        # set defaults
        local_config['parameters']['model']['parameters']['node_features'] = self.dataset.num_node_features()
        local_config['parameters']['model']['parameters']['n_classes'] = self.dataset.num_classes