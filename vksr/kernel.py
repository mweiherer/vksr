from abc import ABC
import numpy as np
import torch


class Kernel(ABC):
    def __init__(self):
        super().__init__()
       
    def _compute_gram_matrix(self, x, y):
        raise NotImplementedError
    
    def __call__(self, x, y):
        return self._compute_gram_matrix(x, y)


class MaternKernel(Kernel):
    def __init__(self, order, h):
        super().__init__()
        self.order = order
        self.h = h
       
    def _compute_gram_matrix(self, x, y):
        d_xy = torch.cdist(x, y)

        if self.order == '1/2':
            return torch.exp(-d_xy / self.h)
        
        if self.order == '3/2':
            return torch.exp(-(np.sqrt(3) * d_xy) / self.h) * (1 + (np.sqrt(3) * d_xy) / self.h) 
       
        if self.order == '5/2':
            return torch.exp(-(np.sqrt(5) * d_xy) / self.h) * (1 + (np.sqrt(5) * d_xy) / self.h + (5 * d_xy ** 2) / (3 * self.h ** 2))
        
        if self.order == 'inf':
            return torch.exp(-d_xy ** 2 / (2 * self.h ** 2))