"""Tensor and array conversion helpers."""

import numpy as np
import torch


def get_torch_device():
    """
    Get the appropriate torch device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_torch_x(x, device=None):
    """
    Transform one sample to a torch tensor of shape (1, n_features).
    """
    if device is None:
        device = get_torch_device()
    if torch.is_tensor(x):
        t = x.detach().clone().to(device=device, dtype=torch.float32).reshape(-1)
    else:
        t = torch.tensor(x, dtype=torch.float32, device=device).reshape(-1)
    return t.unsqueeze(0)


def to_torch_batch_x(X, device=None):
    """
    Transform a batch of samples to a torch tensor of shape (batch, n_features).
    """
    if device is None:
        device = get_torch_device()
    if torch.is_tensor(X):
        return X.detach().clone().to(device=device, dtype=torch.float32)
    return torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32, device=device)


def to_np_x(x):
    """
    Transform one sample to a numpy array of shape (1, n_features).
    """
    if torch.is_tensor(x):
        a = x.detach().cpu().numpy().reshape(-1)
    else:
        a = np.asarray(x, dtype=np.float32).reshape(-1)
    return a.astype(np.float32, copy=False).reshape(1, -1)


def to_np_batch_x(X):
    """
    Transform a batch of samples to a 2D numpy array of shape (batch, n_features).
    """
    if torch.is_tensor(X):
        return X.detach().cpu().numpy()
    return np.asarray(X, dtype=np.float32)


def to_torch_y(y, device=None):
    """
    Transform labels (y) to a torch tensor of shape (batch,).
    """
    if device is None:
        device = get_torch_device()
    if torch.is_tensor(y):
        return y.to(device=device, dtype=torch.long).reshape(-1)
    return torch.as_tensor(y, dtype=torch.long, device=device).reshape(-1)


def to_np_y(y):
    """
    Transform labels (y) to a 1D numpy vector.
    """
    if torch.is_tensor(y):
        return y.detach().cpu().numpy().reshape(-1)
    return np.asarray(y, dtype=np.float32).reshape(-1)


def expectations_to_tensor(z):
    """
    Convert batched PennyLane expectation values to a (batch, n_qubits) tensor.
    Expects a list of n_qubits tensors, each shaped (batch,).
    """
    if torch.is_tensor(z):
        return z.float()
    return torch.stack(z, dim=1).float()