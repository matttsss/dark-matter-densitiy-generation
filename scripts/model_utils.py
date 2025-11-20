import torch

class LinearRegression(torch.nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.W = None

    def fit(self, X, labels):
        X = torch.concatenate([torch.ones(X.shape[0], 1, device=self.device), X], dim=1)
        self.W = (torch.linalg.pinv(X.T @ X) @ X.T @ labels).detach()

        return X @ self.W

    def predict(self, X):
        if self.W is None:
            raise ValueError("Model has not been fitted yet.")
        
        X = torch.concatenate([torch.ones(X.shape[0], 1, device=self.device), X], dim=1)
        return X @ self.W
    


def batch_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list,tuple)):
        return type(batch)(batch_to_device(v, device) for v in batch)
    return batch
