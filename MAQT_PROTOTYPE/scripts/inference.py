"""Algorithm 3: Unified inference with disentangled rejection."""

import numpy as np
import torch

from scripts.constants import DEFAULT_BATCH_SIZE, DEFAULT_CF, DEFAULT_NOISE_RATE, DEFAULT_N_PROBE, DEFAULT_DELTA, ZERO_DAY
from scripts.quantum_metrics import fidelity, trace_distance
from scripts.utils import expectations_to_tensor, to_np_batch_x, to_torch_batch_x, to_torch_x, to_np_y


def predict_labels(X, y, theta, classifier_head, forward_circuit, device, batch_size=DEFAULT_BATCH_SIZE):
    """
    Batched prediction of class labels using circuit expectations + classical head.
    """
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x_chunk = to_torch_batch_x(X[i:i + batch_size], device=device)
            z, _ = forward_circuit(x_chunk, theta)
            logits = classifier_head(expectations_to_tensor(z))
            preds.append(logits.argmax(dim=1).cpu().numpy())
    y_true = to_np_y(y).astype(int)
    return y_true, np.concatenate(preds)


def estimate_lipschitz(X, theta, forward_circuit, n_probe=DEFAULT_N_PROBE, delta=DEFAULT_DELTA, device=None, batch_size=DEFAULT_BATCH_SIZE):
    """
    Batched estimation of Lipschitz constant from data.
    """
    X_np = to_np_batch_x(X)
    if X_np.ndim == 1:
        X_np = X_np.reshape(1, -1)

    n = X_np.shape[0]
    indices = np.random.choice(n, min(n_probe, n), replace=False)

    x_probe = to_torch_batch_x(X_np[indices], device=device)
    x_probe_pert = x_probe + delta * torch.randn_like(x_probe)

    ratios = []
    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            _, r1 = forward_circuit(x_probe[i:i + batch_size], theta)
            _, r2 = forward_circuit(x_probe_pert[i:i + batch_size], theta)
            for j in range(r1.shape[0]):
                ratios.append(trace_distance(r1[j], r2[j]) / delta)

    return float(torch.max(torch.stack(ratios)).item())


def qsnet_infer_single(x, theta, prototypes, q, forward_circuit, p=DEFAULT_NOISE_RATE,
                L_phi=None, Cf=DEFAULT_CF, zero_day=ZERO_DAY, device=None):
    """
    Single-sample unified inference with disentangled rejection.
    """
    if L_phi is None:
        raise ValueError("L_phi must be provided (estimate with estimate_lipschitz)")

    with torch.no_grad():
        _, rho_batch = forward_circuit(to_torch_x(x, device=device), theta)
        rho_x = rho_batch[0]
        class_ids = sorted(prototypes.keys())
        f_map = {c: float(fidelity(rho_x, prototypes[c]).item()) for c in class_ids}
        f_vals = [f_map[c] for c in class_ids]

    c_star = class_ids[int(np.argmax(f_vals))]
    s = 1.0 - f_map[c_star]

    if s > q:
        return zero_day, 0.0, s, f_map

    sorted_f = sorted(f_vals, reverse=True)
    margin = sorted_f[0] - sorted_f[1] if len(sorted_f) > 1 else sorted_f[0]
    radius = margin / (2.0 * (1.0 - p) * L_phi * Cf)

    return c_star, float(radius), s, f_map


def qsnet_infer_batch(X, theta, prototypes, q, forward_circuit, p=DEFAULT_NOISE_RATE,
                       L_phi=None, Cf=DEFAULT_CF, zero_day=ZERO_DAY, device=None, batch_size=DEFAULT_BATCH_SIZE):
    """
    Batched unified inference with disentangled rejection.
    """
    if L_phi is None:
        raise ValueError("L_phi must be provided (estimate with estimate_lipschitz)")

    class_ids = sorted(prototypes.keys())
    labels, radii, scores, f_maps = [], [], [], []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x_chunk = to_torch_batch_x(X[i:i + batch_size], device=device)
            _, rho_chunk = forward_circuit(x_chunk, theta)

            for j in range(rho_chunk.shape[0]):
                rho_x = rho_chunk[j]
                f_map = {c: float(fidelity(rho_x, prototypes[c]).item()) for c in class_ids}
                f_vals = [f_map[c] for c in class_ids]
                c_star = class_ids[int(np.argmax(f_vals))]
                s = 1.0 - f_map[c_star]

                if s > q:
                    labels.append(zero_day); radii.append(0.0)
                else:
                    sorted_f = sorted(f_vals, reverse=True)
                    margin = sorted_f[0] - sorted_f[1] if len(sorted_f) > 1 else sorted_f[0]
                    radius = margin / (2.0 * (1.0 - p) * L_phi * Cf)
                    labels.append(c_star); radii.append(float(radius))

                scores.append(s)
                f_maps.append(f_map)

    return np.array(labels), np.array(radii), np.array(scores), f_maps