"""Algorithm 2: Conformal Zero-Day Threshold Calibration (CQ-ZDR)."""

import numpy as np
import torch

from scripts.quantum_metrics import fidelity
from scripts.utils import to_torch_batch_x
from scripts.constants import DEFAULT_BATCH_SIZE, DEFAULT_ALPHA


def nonconformity_score(X, theta, prototypes, forward_circuit, device=None, batch_size=DEFAULT_BATCH_SIZE):
    """
    Compute the nonconformity scores for a batch of samples.
    """
    scores = []
    class_ids = sorted(prototypes)
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x_chunk = to_torch_batch_x(X[i:i + batch_size], device=device)
            _, rho_chunk = forward_circuit(x_chunk, theta)
            for j in range(rho_chunk.shape[0]):
                fids = [fidelity(rho_chunk[j], prototypes[c]) for c in class_ids]
                scores.append(float((1.0 - torch.max(torch.stack(fids))).item()))
    return np.array(scores)


def calibrate_threshold(theta, X_cal, prototypes, forward_circuit, alpha=DEFAULT_ALPHA, device=None, batch_size=DEFAULT_BATCH_SIZE):
    """
    Calibrate the CQ-ZDR threshold q from the calibration split.
    """
    scores = nonconformity_score(X_cal, theta, prototypes, forward_circuit, device=device, batch_size=batch_size)
    scores_sorted = np.sort(scores)
    n = len(scores_sorted)
    k = int(np.ceil((1.0 - alpha) * (n + 1))) - 1
    k = min(max(k, 0), n - 1) # clamp to [0, n-1]
    q = float(scores_sorted[k])
    return q, scores_sorted