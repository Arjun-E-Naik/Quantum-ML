"""QS-Net PyTorch training and inference modules."""

from scripts.attacks import (
    eval_attacked,
    fgsm_attack,
    logits_from_batch,
    pgd_attack,
    robustness_ablation,
)
from scripts.circuit import (
    build_forward_circuit,
    create_quantum_device,
    initialize_random_weights,
    initialize_zero_weights,
)
from scripts.conformal import calibrate_threshold, nonconformity_score
from scripts.constants import DEFAULT_CF, DEFAULT_NOISE_RATE, ZERO_DAY
from scripts.data import (
    balanced_sample,
    capped_sample,
    class_balance_table,
    class_weights_for_sampler,
    load_split,
    plot_class_balance_bars,
    plot_class_balance_pie,
    stratified_head,
)
from scripts.gradient import gradient_variance, gradient_variance_probe
from scripts.hilbert import fidelity_gap_proxy, hilbert_geometry_diagnostics, print_h1_report
from scripts.inference import (
    estimate_lipschitz,
    predict_labels,
    qsnet_infer_batch,
    qsnet_infer_single,
)
from scripts.logging import to_jsonable, write_history_log, write_sweep_log
from scripts.loss import (
    ce_loss_term,
    compute_l_ce,
    compute_l_inter,
    compute_l_intra,
    curriculum_weight,
    focal_ce,
    inter_loss_term,
    intra_loss_term,
    maqt_loss,
)
from scripts.prototypes import (
    EMAPrototypeBank,
    PrototypeBank,
    prototype_summary,
)
from scripts.quantum_metrics import fidelity, trace_distance
from scripts.sweep import default_score_fn, run_sweep
from scripts.utils import (
    expectations_to_tensor,
    get_torch_device,
    to_np_batch_x,
    to_np_x,
    to_np_y,
    to_torch_batch_x,
    to_torch_x,
    to_torch_y,
)

__all__ = [
    "DEFAULT_CF",
    "DEFAULT_NOISE_RATE",
    "EMAPrototypeBank",
    "PrototypeBank",
    "ZERO_DAY",
    "balanced_sample",
    "build_forward_circuit",
    "calibrate_threshold",
    "capped_sample",
    "ce_loss_term",
    "class_balance_table",
    "class_weights_for_sampler",
    "compute_l_ce",
    "compute_l_inter",
    "compute_l_intra",
    "create_quantum_device",
    "curriculum_weight",
    "default_score_fn",
    "estimate_lipschitz",
    "eval_attacked",
    "expectations_to_tensor",
    "fgsm_attack",
    "fidelity",
    "fidelity_gap_proxy",
    "focal_ce",
    "get_torch_device",
    "gradient_variance",
    "gradient_variance_probe",
    "hilbert_geometry_diagnostics",
    "initialize_random_weights",
    "initialize_zero_weights",
    "inter_loss_term",
    "intra_loss_term",
    "load_split",
    "logits_from_batch",
    "maqt_loss",
    "nonconformity_score",
    "pgd_attack",
    "plot_class_balance_bars",
    "plot_class_balance_pie",
    "predict_labels",
    "print_h1_report",
    "prototype_summary",
    "qsnet_infer_batch",
    "qsnet_infer_single",
    "robustness_ablation",
    "run_sweep",
    "stratified_head",
    "to_jsonable",
    "to_np_batch_x",
    "to_np_x",
    "to_np_y",
    "to_torch_batch_x",
    "to_torch_x",
    "to_torch_y",
    "trace_distance",
    "write_history_log",
    "write_sweep_log",
]