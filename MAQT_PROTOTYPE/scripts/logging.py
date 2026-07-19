"""Training history logging helpers."""

import json
from pathlib import Path

from scripts.constants import DEFAULT_LOG_DIR, DEFAULT_SWEEP_LOG_DIR


def to_jsonable(obj):
    """
    Convert nested `history` metrics into JSON-serializable values.
    """
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    return str(obj)


def write_history_log(history, name, extra=None, log_dir=DEFAULT_LOG_DIR):
    """
    Write training `history` to `<notebook_name>.log` as JSON.
    """
    log_path = Path(log_dir) / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"history": to_jsonable(history)}
    if extra:
        payload["extra"] = to_jsonable(extra)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return log_path


def write_sweep_log(sweep_df, name, log_dir=DEFAULT_SWEEP_LOG_DIR):
    """
    Write a hyperparameter sweep DataFrame (from `sweep.py`) to `<sweep_name>.json` as JSON.
    """
    log_path = Path(log_dir) / f"{name}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    payload = to_jsonable(sweep_df.to_dict(orient="records"))

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return log_path