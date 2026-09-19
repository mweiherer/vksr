from abc import ABC
import torch


class KernelSolver(ABC):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel

    def fit_and_predict(self, x, y, reg_weight, x_query):
        raise NotImplementedError

 
class CholeskyKernelSolver(KernelSolver):
    def __init__(self, kernel):
        super().__init__(kernel)

        self.jitter_32 = 1e-5
        self.jitter_64 = 1e-13
    
    @torch.compile()
    def fit_and_predict(self, x, y, reg_weight, x_query):
        reg_weight2 = x.shape[1] * reg_weight + self.jitter_32 if x.dtype == torch.float32 else self.jitter_64

        G = self.kernel(x, x)
        A = G + reg_weight2 * torch.eye(G.shape[1]).repeat(G.shape[0], 1, 1).to(G)

        L, info = torch.linalg.cholesky_ex(A)

        if torch.any(info > 0):
            print(f'Cholesky decomposition failed for {torch.sum(info > 0)} system(s). Please increase jitter or regularization.')

        return self.kernel(x_query, x) @ torch.cholesky_solve(y, L)