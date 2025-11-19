from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn


class Perturber(nn.Module, metaclass=ABCMeta):
    """
    Base class for graph perturbation models.
    """
    
    def __init__(self, cfg, model: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self._model_active = True
    
    def deactivate_model(self):
        """Deactivate the model (e.g., set to eval mode, disable gradients)."""
        self._model_active = False
        if hasattr(self.model, 'eval'):
            self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    def activate_model(self):
        """Activate the model (e.g., set to train mode, enable gradients)."""
        self._model_active = True
        if hasattr(self.model, 'train'):
            self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass through the perturber."""
        pass

