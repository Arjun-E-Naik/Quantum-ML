"""
MAQT training loop: batched circuit, EMA prototypes, curriculum lambda, optional focal loss, weighted sampling.
"""

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler

from scripts.circuit import initialize_random_weights
from scripts.constants import (
    BARREN_PLATEAU_VAR_THRESHOLD,
    DEFAULT_CONTROL_BATCH_SIZE,
    DEFAULT_EMA_MOMENTUM,
    DEFAULT_EPOCHS,
    DEFAULT_FOCAL,
    DEFAULT_FOCAL_GAMMA,
    DEFAULT_GRAD_CLIP_NORM,
    DEFAULT_LAMBDA1,
    DEFAULT_LAMBDA2,
    DEFAULT_LR,
    DEFAULT_WARMUP_FRAC,
    DEFAULT_WEIGHT_INIT_EPS,
)
from scripts.data import class_weights_for_sampler
from scripts.loss import curriculum_weight, maqt_loss
from scripts.prototypes import EMAPrototypeBank, PrototypeBank
from scripts.quantum_metrics import trace_distance
from scripts.utils import expectations_to_tensor, to_np_batch_x, to_np_y, to_torch_batch_x, to_torch_y


def train_maqt(
    X_train,
    y_train,
    n_classes,
    n_qubits,
    n_layers,
    forward_circuit,
    device,
    epochs=DEFAULT_EPOCHS,
    lr=DEFAULT_LR,
    batch_size=DEFAULT_CONTROL_BATCH_SIZE,
    lambda1_max=DEFAULT_LAMBDA1,
    lambda2_max=DEFAULT_LAMBDA2,
    warmup_frac=DEFAULT_WARMUP_FRAC,
    grad_clip_norm=DEFAULT_GRAD_CLIP_NORM,
    ema_momentum=DEFAULT_EMA_MOMENTUM,
    use_focal=DEFAULT_FOCAL,
    focal_gamma=DEFAULT_FOCAL_GAMMA,
    weight_init_eps=DEFAULT_WEIGHT_INIT_EPS,
    use_weighted_sampler=True,
    log_every=1,
    verbose=True,
):
    """
    Train the MAQT model.
    """
    theta = initialize_random_weights(n_layers, n_qubits, device, eps=weight_init_eps)
    head = nn.Linear(n_qubits, n_classes).to(device)
    opt = torch.optim.Adam(list([theta]) + list(head.parameters()), lr=lr)
    ce_loss_fn = nn.CrossEntropyLoss()

    ema_protos = EMAPrototypeBank(classes=range(n_classes), momentum=ema_momentum)

    X_t = to_torch_batch_x(X_train, device=device)
    y_t = to_torch_y(y_train, device=device)

    if use_weighted_sampler:
        weights = class_weights_for_sampler(to_np_y(y_train), n_classes)
        sample_weights = torch.as_tensor(weights, dtype=torch.double)
    else:
        sample_weights = None

    history = {
        "loss": [],
        "L_CE": [],
        "L_intra": [],
        "L_inter": [],
        "grad_var": [],
        "intra_fid_gap": [],
        "inter_trace_dist": [],
        "epoch_sec": [],
    }

    n = len(X_t)
    for epoch in range(epochs):
        epoch_t0 = time.perf_counter()

        lam1 = curriculum_weight(epoch, epochs, lambda1_max, warmup_frac)
        lam2 = curriculum_weight(epoch, epochs, lambda2_max, warmup_frac)

        if use_weighted_sampler:
            perm = torch.tensor(
                list(WeightedRandomSampler(sample_weights, num_samples=n, replacement=True))
            )
        else:
            perm = torch.randperm(n)

        epoch_terms = {"L_total": [], "L_CE": [], "L_intra": [], "L_inter": []}
        grad_vars, intra_fid_running = [], []

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X_t[idx], y_t[idx]

            protos_snapshot = ema_protos.snapshot()
            if not protos_snapshot:
                # first steps before any class seen: CE only (lambda terms are 0)
                protos_snapshot = {}

            loss, l_ce, l_intra, l_inter, y_used, rho_used = maqt_loss(
                theta, head, ce_loss_fn, xb, yb, protos_snapshot, forward_circuit,
                lambda1=lam1, lambda2=lam2, device=device,
                use_focal=use_focal, focal_gamma=focal_gamma,
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list([theta]) + list(head.parameters()), grad_clip_norm)
            grad_vars.append(theta.grad.var().item())
            opt.step()

            # EMA update strictly after loss/backward (see prototypes.py)
            batch_means = ema_protos.batch_class_means(rho_used.detach(), y_used)
            ema_protos.update(batch_means)

            epoch_terms["L_total"].append(loss.item())
            epoch_terms["L_CE"].append(l_ce.item())
            epoch_terms["L_intra"].append(l_intra.item())
            epoch_terms["L_inter"].append(l_inter.item())
            intra_fid_running.append(1 - l_intra.item())

        diag_pairs = []
        keys = sorted(ema_protos.protos.keys())
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                diag_pairs.append(float(trace_distance(ema_protos.protos[keys[a]], ema_protos.protos[keys[b]])))
        mean_inter_td = float(np.mean(diag_pairs)) if diag_pairs else 0.0
        mean_gv = float(np.mean(grad_vars))
        epoch_sec = time.perf_counter() - epoch_t0

        history["loss"].append(float(np.mean(epoch_terms["L_total"])))
        history["L_CE"].append(float(np.mean(epoch_terms["L_CE"])))
        history["L_intra"].append(float(np.mean(epoch_terms["L_intra"])))
        history["L_inter"].append(float(np.mean(epoch_terms["L_inter"])))
        history["grad_var"].append(mean_gv)
        history["intra_fid_gap"].append(float(np.mean(intra_fid_running)))
        history["inter_trace_dist"].append(mean_inter_td)
        history["epoch_sec"].append(epoch_sec)

        if verbose and (epoch + 1) % log_every == 0:
            print(
                f"epoch {epoch+1:2d}/{epochs} | loss {history['loss'][-1]:.4f} | "
                f"L_CE {history['L_CE'][-1]:.4f} | L_intra {history['L_intra'][-1]:.4f} | "
                f"L_inter {history['L_inter'][-1]:.4f} | grad_var {mean_gv:.2e} | "
                f"intra_fid {history['intra_fid_gap'][-1]:.3f} | inter_TD {mean_inter_td:.3f} | "
                f"time {epoch_sec:.1f}s"
            )
            if mean_gv < BARREN_PLATEAU_VAR_THRESHOLD:
                print("  barren plateau detected")

    # final exact unbiased prototypes (frozen theta) for deployment
    final_bank = PrototypeBank(classes=range(n_classes))
    final_prototypes = final_bank.compute(
        theta,
        to_np_batch_x(X_train),
        to_np_y(y_train),
        forward_circuit=forward_circuit,
        device=device,
    )

    return theta, head, final_prototypes, ema_protos, history


def train_plain_vqc(
    X_train,
    y_train,
    n_classes,
    n_qubits,
    n_layers,
    forward_circuit,
    device,
    epochs=DEFAULT_EPOCHS,
    lr=DEFAULT_LR,
    batch_size=DEFAULT_CONTROL_BATCH_SIZE,
    weight_init_eps=DEFAULT_WEIGHT_INIT_EPS,
    grad_clip_norm=DEFAULT_GRAD_CLIP_NORM,
):
    """CE-only baseline (no prototype terms) for robustness ablation."""
    theta = initialize_random_weights(n_layers, n_qubits, device, eps=weight_init_eps)
    head = nn.Linear(n_qubits, n_classes).to(device)
    opt = torch.optim.Adam(list([theta]) + list(head.parameters()), lr=lr)
    ce = nn.CrossEntropyLoss()

    X_t = to_torch_batch_x(X_train, device=device)
    y_t = to_torch_y(y_train, device=device)
    n = len(X_t)
    losses = []

    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X_t[idx], y_t[idx]
            z, _ = forward_circuit(xb, theta)
            logits = head(expectations_to_tensor(z))
            loss = ce(logits, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list([theta]) + list(head.parameters()), grad_clip_norm
            )
            opt.step()
        losses.append(loss.item())
        print(f"[plain VQC] epoch {epoch+1}/{epochs} loss {loss.item():.4f}")
    return theta, head, losses