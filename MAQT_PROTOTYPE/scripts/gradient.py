"""Gradient variance helpers."""

from pennylane import numpy as np
import torch

from scripts.circuit import initialize_random_weights
from scripts.constants import DEFAULT_LAMBDA1, DEFAULT_LAMBDA2, DEFAULT_N_TRIALS, DEFAULT_WEIGHT_INIT_EPS
from scripts.loss import maqt_loss
from scripts.utils import get_torch_device, to_torch_batch_x


def gradient_variance_probe(forward_circuit, x_probe, n_qubits, n_layers, 
                            n_trials=DEFAULT_N_TRIALS, eps=DEFAULT_WEIGHT_INIT_EPS, device=None):
    """
    Pre-training barren-plateau check to catch barren-plateau architectures cheaply.

    Runs `n_trials` fresh small-random inits through the circuit on a fixed probe batch,
    backprops a purity-based toy loss, and
    reports the variance of a single gradient component across inits.
    """
    device = device or get_torch_device()
    x_probe = to_torch_batch_x(x_probe, device=device)
    grad_samples = []
    for _ in range(n_trials):
        theta = initialize_random_weights(n_layers, n_qubits, device, eps=eps)
        _, rho = forward_circuit(x_probe, theta)
        loss = torch.stack([torch.real(torch.trace(r @ r)) for r in rho]).mean()
        loss.backward()
        grad_samples.append(theta.grad.reshape(-1)[0].item())
    grad_samples = np.asarray(grad_samples, dtype=np.float64)
    var = float(grad_samples.var())
    return var, grad_samples


def gradient_variance(theta, classifier_head, ce_loss_fn, X_batch, y_batch, prototypes, forward_circuit,
                      lambda1=DEFAULT_LAMBDA1, lambda2=DEFAULT_LAMBDA2, device=None):
    """
    Fresh grad of MAQT loss w.r.t. circuit theta, then variance over parameters.
    """
    device = device or theta.device
    loss, *_ = maqt_loss(theta, classifier_head, ce_loss_fn, X_batch, y_batch, prototypes,
                                forward_circuit, lambda1=lambda1, lambda2=lambda2, device=device)
    grads = torch.autograd.grad(loss, theta, retain_graph=False, create_graph=False)[0]
    flat = grads.reshape(-1)
    return float(flat.var().item()), grads