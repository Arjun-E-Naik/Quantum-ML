"""Algorithm 1: Margin-Aware Quantum Training (MAQT) loss."""

import torch
import torch.nn.functional as F

from scripts.quantum_metrics import fidelity, trace_distance
from scripts.utils import expectations_to_tensor, to_torch_batch_x, to_torch_y, to_np_y
from scripts.constants import DEFAULT_FOCAL, DEFAULT_FOCAL_GAMMA, DEFAULT_LAMBDA1, DEFAULT_LAMBDA2, DEFAULT_WARMUP_FRAC


def _forward_batch(theta, X_batch, y_batch, forward_circuit, device):
    """
    Run the circuit on a batch; return (y_t, z, rho).
    """
    x_t = to_torch_batch_x(X_batch, device=device)
    y_t = to_torch_y(y_batch, device=device)
    z, rho = forward_circuit(x_t, theta)
    return y_t, z, rho


def focal_ce(logits, targets, gamma=DEFAULT_FOCAL_GAMMA):
    """
    Focal loss, an optional alternative to plain cross-entropy.
    """
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def ce_loss_term(y_t, z, classifier_head, ce_loss_fn, device, use_focal=DEFAULT_FOCAL, focal_gamma=DEFAULT_FOCAL_GAMMA):
    """
    Cross-entropy or focal loss term (L_CE) from batched circuit expectations + linear head.
    """
    if y_t.numel() == 0:
        return torch.tensor(0.0, device=device)
    logits = classifier_head(expectations_to_tensor(z))  # (batch, n_classes)
    if use_focal:
        return focal_ce(logits, y_t, gamma=focal_gamma)
    return ce_loss_fn(logits, y_t)


def intra_loss_term(y_t, rho, prototypes, device):
    """
    Intra-class term (L_intra): mean infidelity (1 - F) to each class prototype.
    """
    y_np = to_np_y(y_t)
    intra_terms = {c: [] for c in prototypes}
    for i, c_val in enumerate(y_np):
        c = int(c_val)
        if c not in prototypes:
            continue
        intra_terms[c].append(1.0 - fidelity(rho[i], prototypes[c]))

    l_intra = torch.tensor(0.0, device=device)
    active_classes = [c for c, terms in intra_terms.items() if terms]
    for c in active_classes:
        l_intra = l_intra + torch.stack(intra_terms[c]).mean()
    if active_classes:
        l_intra = l_intra / len(active_classes)
    return l_intra


def inter_loss_term(prototypes, device):
    """
    Inter-class term (L_inter): negative mean trace distance between prototypes.
    """
    class_ids = sorted(prototypes.keys())
    inter_pairs = []
    for i, c in enumerate(class_ids):
        for c_prime in class_ids[i + 1 :]:
            inter_pairs.append(trace_distance(prototypes[c], prototypes[c_prime]))
    if not inter_pairs:
        return torch.tensor(0.0, device=device)
    return -torch.stack(inter_pairs).mean()


def compute_l_ce(theta, classifier_head, ce_loss_fn, X_batch, y_batch, forward_circuit, device=None,
                  use_focal=DEFAULT_FOCAL, focal_gamma=DEFAULT_FOCAL_GAMMA):
    """
    Unit-testable L_CE over a batch.
    """
    device = device or theta.device
    y_t, z, _ = _forward_batch(theta, X_batch, y_batch, forward_circuit, device)
    return ce_loss_term(y_t, z, classifier_head, ce_loss_fn, device, use_focal, focal_gamma)


def compute_l_intra(theta, X_batch, y_batch, prototypes, forward_circuit, device=None):
    """
    Unit-testable L_intra over a batch.
    """
    device = device or theta.device
    y_t, _, rho = _forward_batch(theta, X_batch, y_batch, forward_circuit, device)
    return intra_loss_term(y_t, rho, prototypes, device)


def compute_l_inter(prototypes, device=None):
    """
    Unit-testable L_inter from class prototypes only.
    """
    if device is None:
        any_rho = next(iter(prototypes.values()))
        device = any_rho.device
    return inter_loss_term(prototypes, device)


def maqt_loss(theta, classifier_head, ce_loss_fn, X_batch, y_batch, prototypes,
            forward_circuit, lambda1=DEFAULT_LAMBDA1, lambda2=DEFAULT_LAMBDA2, device=None,
            use_focal=DEFAULT_FOCAL, focal_gamma=DEFAULT_FOCAL_GAMMA):
    """
    MAQT loss: L = L_CE + lambda1 * L_intra + lambda2 * L_inter.
    """
    device = device or theta.device
    y_t, z, rho = _forward_batch(theta, X_batch, y_batch, forward_circuit, device)

    l_ce = ce_loss_term(y_t, z, classifier_head, ce_loss_fn, device, use_focal, focal_gamma)
    l_intra = intra_loss_term(y_t, rho, prototypes, device)
    l_inter = inter_loss_term(prototypes, device)

    loss = l_ce + lambda1 * l_intra + lambda2 * l_inter
    return loss, l_ce, l_intra, l_inter, y_t, rho


def curriculum_weight(epoch, total_epochs, base, warmup_frac=DEFAULT_WARMUP_FRAC):
    """
    Linearly ramp a lambda from 0 to `base` over the first `warmup_frac` of training.

    This prevents the circuit from being prematurely constrained by uninformative prototypes during early training.
    """
    warmup = max(int(warmup_frac * total_epochs), 1)
    return base * min(1.0, epoch / warmup)