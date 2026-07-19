"""FGSM / PGD adversarial attacks on the classical feature input."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from scripts.utils import expectations_to_tensor, to_torch_batch_x, to_torch_y, to_np_y
from scripts.constants import DEFAULT_BATCH_SIZE, DEFAULT_CONTROL_BATCH_SIZE


def logits_from_batch(X, theta, classifier_head, forward_circuit):
    """
    Compute logits from a batch of inputs.
    """
    z, _ = forward_circuit(X, theta)
    return classifier_head(expectations_to_tensor(z))


def fgsm_attack(X, y, theta, classifier_head, forward_circuit, eps, device,
                x_min=None, x_max=None, ce_loss_fn=None):
    """
    Batched one-step FGSM.
    Perturbs features by eps along the sign of the CE loss gradient.
    """
    if ce_loss_fn is None:
        ce_loss_fn = nn.CrossEntropyLoss()

    X_t = to_torch_batch_x(X, device=device).detach().requires_grad_(True)
    y_t = to_torch_y(y, device=device)

    logits = logits_from_batch(X_t, theta, classifier_head, forward_circuit)
    if getattr(ce_loss_fn, "weight", None) is None:
        loss = ce_loss_fn(logits, y_t)
    else:
        loss = nn.CrossEntropyLoss()(logits, y_t)   # attacks use unweighted CE

    grad = torch.autograd.grad(loss, X_t)[0]

    X_adv = X_t + eps * grad.sign()
    if x_min is not None:
        X_adv = torch.maximum(X_adv, x_min)
    if x_max is not None:
        X_adv = torch.minimum(X_adv, x_max)
    return X_adv.detach()


def pgd_attack(X, y, theta, classifier_head, forward_circuit, eps, alpha, steps,
               device, x_min=None, x_max=None, ce_loss_fn=None, random_start=True):
    """
    Batched multi-step PGD with L_inf projection onto the eps-ball around X.
    Perturbs features by alpha along the sign of the CE loss gradient.
    """
    if ce_loss_fn is None:
        ce_loss_fn = nn.CrossEntropyLoss()
    X0 = to_torch_batch_x(X, device=device).detach()
    y_t = to_torch_y(y, device=device)

    if random_start:
        X_adv = X0 + torch.empty_like(X0).uniform_(-eps, eps)
    else:
        X_adv = X0.clone()
    if x_min is not None:
        X_adv = torch.maximum(X_adv, x_min)
    if x_max is not None:
        X_adv = torch.minimum(X_adv, x_max)
    X_adv = X_adv.detach()

    for _ in range(steps):
        X_adv.requires_grad_(True)
        logits = logits_from_batch(X_adv, theta, classifier_head, forward_circuit)
        loss = ce_loss_fn(logits, y_t) if getattr(ce_loss_fn, "weight", None) is None \
            else nn.CrossEntropyLoss()(logits, y_t)
        grad = torch.autograd.grad(loss, X_adv)[0]

        with torch.no_grad():
            X_adv = X_adv + alpha * grad.sign()
            X_adv = torch.maximum(torch.minimum(X_adv, X0 + eps), X0 - eps)
            if x_min is not None:
                X_adv = torch.maximum(X_adv, x_min)
            if x_max is not None:
                X_adv = torch.minimum(X_adv, x_max)
        X_adv = X_adv.detach()
    return X_adv


def eval_attacked(attack_fn, X, y, theta, classifier_head, forward_circuit, device,
                  labels=None, batch_size=DEFAULT_BATCH_SIZE, **attack_kwargs):
    """
    Run an attack over a dataset and report accuracy + macro-F1.
    """
    y_true = to_np_y(y).astype(int)
    preds = []
    with torch.enable_grad():
        for i in range(0, len(X), batch_size):
            X_chunk = X[i:i + batch_size]
            y_chunk = y_true[i:i + batch_size]

            X_adv = attack_fn(X_chunk, y_chunk, theta, classifier_head, forward_circuit,
                               device=device, **attack_kwargs)
            with torch.no_grad():
                logits = logits_from_batch(X_adv, theta, classifier_head, forward_circuit)
                preds.append(logits.argmax(dim=1).cpu().numpy())

    preds = np.concatenate(preds)
    return {
        "acc": float(np.mean(y_true == preds)),
        "macro_f1": float(
            f1_score(
                y_true,
                preds,
                average="macro",
                labels=labels,
                zero_division=0,
            )
        ),
        "y_true": y_true,
        "y_pred": preds,
    }


def train_plain_baseline(X_train, y_train, n_classes, forward_circuit, n_qubits,
                        epochs, lr, device, batch_size=DEFAULT_CONTROL_BATCH_SIZE):
    """
    Plain-CE control model for FGSM/PGD ablation (no prototype / MAQT losses).
    """
    # from scripts.circuit import initialize_random_weights

    # theta = initialize_random_weights(n_layers=None, n_wires=n_qubits,
    #                                     device=device, eps=DEFAULT_WEIGHT_INIT_EPS)
    
    # raise NotImplementedError(
    #     "Wire this to your project's theta/head init (see train.py's train_maqt for the pattern); "
    #     "kept as a stub here since layer count isn't known inside attacks.py."
    # )
    pass


def robustness_ablation(X_test, y_test, theta_maqt, head_maqt, theta_plain, head_plain,
                         forward_circuit, epsilons, device, attack_fn=fgsm_attack,
                         batch_size=DEFAULT_BATCH_SIZE, **attack_kwargs):
    """
    Compares MAQT-trained model vs. plain-CE baseline under the same attack at each epsilon.
    """
    results = []
    for eps in epsilons:
        maqt_res = eval_attacked(attack_fn, X_test, y_test, theta_maqt, head_maqt,
                                    forward_circuit, device, batch_size=batch_size,
                                    eps=eps, **attack_kwargs)
        plain_res = eval_attacked(attack_fn, X_test, y_test, theta_plain, head_plain,
                                    forward_circuit, device, batch_size=batch_size,
                                    eps=eps, **attack_kwargs)
        results.append({
            "eps": eps,
            "maqt_acc": maqt_res["acc"],
            "maqt_macro_f1": maqt_res["macro_f1"],
            "plain_acc": plain_res["acc"],
            "plain_macro_f1": plain_res["macro_f1"],
        })
    return results