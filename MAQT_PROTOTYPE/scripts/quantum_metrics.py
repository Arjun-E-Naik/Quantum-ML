"""Quantum state distance and similarity metrics."""

import pennylane as qp
import torch

from scripts.constants import DEFAULT_EPS


def fidelity(rho_a, rho_b, eps=DEFAULT_EPS):
    """
    Fidelity between two density matrices, clamped to [0, 1-eps] to avoid numerical instability.
    """
    f = qp.math.fidelity(rho_a, rho_b)
    if torch.is_tensor(f):
        f = f.real if torch.is_complex(f) else f
        return torch.clamp(f, 0.0, 1.0 - eps)
    return min(max(float(f), 0.0), 1.0 - eps)


def trace_distance(rho_a, rho_b):
    """
    Trace distance between two density matrices, clamped to [0, 1].
    """
    td = qp.math.trace_distance(rho_a, rho_b)
    if torch.is_tensor(td):
        td = td.real if torch.is_complex(td) else td
        return torch.clamp(td, 0.0, 1.0)
    return min(max(float(td), 0.0), 1.0)