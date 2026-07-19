"""Hyperparameter search helpers."""

import pandas as pd

from scripts.constants import BARREN_PLATEAU_VAR_THRESHOLD


def run_sweep(train_fn, param_grid, score_fn):
    """
    Generalized hyperparameter sweep across a parameter grid.

    Returns a DataFrame with the scores, final loss, final intra fidelity gap,
    and minimum gradient variance for each parameter set.
    """
    records = []
    for params in param_grid:
        theta, head, prototypes, ema_protos, history = train_fn(**params)
        score = score_fn(history, prototypes)
        records.append({**params, "score": score, "final_loss": history["loss"][-1],
                         "final_intra_fid": history["intra_fid_gap"][-1],
                         "min_grad_var": min(history["grad_var"])})
        print(f"{params} -> score={score:.4f}")
    df = pd.DataFrame(records).sort_values("score", ascending=False)
    return df


def default_score_fn(history, prototypes):
    """
    Score function: separation + tightness - instability penalty.

    If the gradient variance is below the barren-plateau threshold,
    the instability penalty is 0. Otherwise, it is 1.
    """
    stable = min(history["grad_var"]) > BARREN_PLATEAU_VAR_THRESHOLD
    return history["inter_trace_dist"][-1] + history["intra_fid_gap"][-1] - (0 if stable else 1)