"""Algorithm 1: Class prototype bank and prototype computation for MAQT."""

import torch

from scripts.utils import to_torch_batch_x, to_torch_y, to_np_y
from scripts.constants import DEFAULT_BATCH_SIZE, DEFAULT_EMA_MOMENTUM


class PrototypeBank:
    """
    Exact class-means (prototypes) of rho under frozen theta.

    For the final, unbiased prototypes.
    """

    def __init__(self, classes):
        self.classes = sorted(classes)
        self.protos = {}

    def compute(self, theta, X, y, forward_circuit, device=None, batch_size=DEFAULT_BATCH_SIZE):
        """Fill self.protos with exact class means. Returns self.protos."""
        self.protos = {}
        y_labels = to_np_y(y).astype(int)
        with torch.no_grad():
            for c in self.classes:
                mask = y_labels == c
                if not mask.any():
                    continue
                X_c = X[mask]
                proto_sum = None
                count = 0
                for i in range(0, len(X_c), batch_size):
                    x_chunk = to_torch_batch_x(X_c[i:i + batch_size], device=device)
                    _, rho_chunk = forward_circuit(x_chunk, theta)
                    chunk_sum = rho_chunk.sum(dim=0)

                    if proto_sum is None:
                        proto_sum = chunk_sum.detach().clone()
                    else:
                        proto_sum = proto_sum + chunk_sum.detach()

                    count += len(x_chunk)

                    del rho_chunk, chunk_sum    # free memory

                self.protos[c] = proto_sum / count
                del proto_sum   # free memory

        return self.protos


class EMAPrototypeBank:
    """
    Note: EMA-target methods such as MoCo and DINO avoid using a same-step updated target,
    instead employing a delayed target to reduce the risk of representational collapse.

    Exponential-moving-average class-means (prototypes) of rho(x), updated every minibatch. 
    
    For the live supervision signal inside the training loop's loss.
    """

    def __init__(self, classes, momentum=DEFAULT_EMA_MOMENTUM):
        self.classes = sorted(classes)
        self.momentum = momentum
        self.protos = {}

    def batch_class_means(self, rho_batch, y_batch):
        """Per-class mean of rho over this batch."""
        y_batch = to_torch_y(y_batch, device=rho_batch.device)
        means = {}
        for c in torch.unique(y_batch):
            c_int = int(c)
            mask = y_batch == c
            means[c_int] = rho_batch[mask].mean(dim=0)
        return means

    def snapshot(self):
        """Copy of current prototypes BEFORE a batch's update."""
        return {c: p.clone() for c, p in self.protos.items()}

    def update(self, batch_means):
        """Update prototypes AFTER computing the loss for a batch."""
        for c_int, batch_mean in batch_means.items():
            if c_int not in self.protos:
                self.protos[c_int] = batch_mean.detach().clone()
            else:
                self.protos[c_int] = (
                    self.momentum * self.protos[c_int]
                    + (1 - self.momentum) * batch_mean.detach()
                )
        return self.protos


def prototype_summary(prototypes, class_names=None):
    """
    Summarize the prototype density matrices.
    """
    rows = []
    for c in sorted(prototypes):
        rho_c = prototypes[c]
        name = class_names[c] if class_names is not None else str(c)
        rows.append({
            "class": name,
            "dim": int(rho_c.shape[0]),
            "trace": float(torch.trace(rho_c).real.item()),
        })
    return rows