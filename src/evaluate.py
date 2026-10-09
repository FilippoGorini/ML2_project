"""Metrics for the 4 per-leg binary problems (contact = positive class)."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from src.data import LEGS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "results"


def scores(y, p):
    """y, p: (M, 4) 0/1 arrays. Per-leg F1/precision/recall/accuracy, their mean, exact-state accuracy."""
    per_leg = {
        leg: {
            "f1": f1_score(y[:, k], p[:, k], zero_division=0),
            "precision": precision_score(y[:, k], p[:, k], zero_division=0),
            "recall": recall_score(y[:, k], p[:, k], zero_division=0),
            "accuracy": np.mean(y[:, k] == p[:, k]),
        }
        for k, leg in enumerate(LEGS)
    }
    mean = {m: np.mean([per_leg[leg][m] for leg in LEGS]) for m in per_leg[LEGS[0]]}
    return {**mean, "exact": np.mean((y == p).all(1)), "per_leg": per_leg}


def report(name, y_test, p_test, validation=()):
    """Print and save test scores on every timestep, and on every 10th (stride-10 check).
    Also saves the validation scores of the hyperparameter search and the test predictions."""
    full, sub = scores(y_test, p_test), scores(y_test[::10], p_test[::10])
    print(f"\n{name} — test, every timestep (n={len(y_test)})")
    print(f"{'':6s}{'F1':>8s}{'prec':>8s}{'recall':>8s}{'acc':>8s}")
    for leg in LEGS + ["mean"]:
        s = full["per_leg"][leg] if leg in LEGS else full
        print(f"{leg:6s}" + "".join(f"{s[m]:8.4f}" for m in ("f1", "precision", "recall", "accuracy")))
    print(f"exact-state accuracy {full['exact']:.4f}")
    print(f"every 10th timestep: F1 {sub['f1']:.4f}, exact-state {sub['exact']:.4f}")

    (RESULTS_DIR / name).parent.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{name}.json").write_text(
        json.dumps({"test": full, "test_stride10": sub, "validation": list(validation)}, indent=2))
    np.save(RESULTS_DIR / f"{name}_test_pred.npy", p_test.astype(np.uint8))


if __name__ == "__main__":
    y = np.array([[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 0, 0]])
    s = scores(y, y)
    assert s["f1"] == 1 and s["exact"] == 1
    s = scores(y, np.zeros_like(y))  # "never contact": F1 0, accuracy = share of zeros
    assert s["f1"] == 0 and s["recall"] == 0 and s["accuracy"] == np.mean(y == 0)
    p = y.copy(); p[0, 0] = 0  # one missed contact on RF
    s = scores(y, p)
    assert s["per_leg"]["RF"]["recall"] == 0.5 and s["per_leg"]["RF"]["precision"] == 1 and s["exact"] == 0.75
    print("evaluate.py self-check OK")
