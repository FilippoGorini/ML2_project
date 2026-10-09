"""Tune a model on validation, refit on train+val, report test scores.

    python -m src.train majority
    python -m src.train ridge current
    python -m src.train ridge window_log --val-only    # tune and score on validation only, no test
    python -m src.train ridge feat_all --stride=150    # train/val stride 150 (non-overlapping windows);
                                                       # saved under stride150/ (default stride 10)
    python -m src.train ridge feat_all --protocol-b    # unseen recordings (src/data.py); saved under protocol_b/
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from src.data import labels, load_recordings, split, split_b
from src.evaluate import RESULTS_DIR, report
from src.features import INPUTS
from src.models import MODELS

MODELS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "models"
CHUNK = 20_000  # test windows predicted at a time (kernel models / wide inputs would not fit otherwise)


def main(model_name, input_name="none", val_only=False, stride=10, protocol_b=False):
    name = model_name if input_name == "none" else f"{model_name}_{input_name}"
    if stride != 10:
        name = f"stride{stride}/{name}"
    if protocol_b:
        name = f"protocol_b/{name}"
    recs = load_recordings()
    sp = (split_b if protocol_b else split)(recs, stride=stride)
    make_input, (make_model, grid) = INPUTS[input_name], MODELS[model_name]
    n_train = sum(len(e) for e in sp["train"].values())
    X = np.concatenate([make_input(recs, sp["train"]), make_input(recs, sp["val"])])
    y = np.concatenate([labels(recs, sp["train"]), labels(recs, sp["val"])])

    # One fixed train -> val fold (no random k-fold: neighbouring windows are near-duplicates).
    # Score = macro per-leg F1. After choosing, refit on train+val and test once.
    search = GridSearchCV(
        make_model(), grid,
        cv=PredefinedSplit(np.r_[np.full(n_train, -1), np.zeros(len(X) - n_train)]),
        scoring="f1_macro", refit=not val_only,
        n_jobs=2,  # each parallel fit copies the training data; more workers can run out of RAM on wide inputs
    ).fit(X, y)
    del X, y

    validation = [{"params": p, "val_f1": s}
                  for p, s in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"])]
    for v in validation:
        print(f"  {v['params']}  val F1 {v['val_f1']:.4f}")
    print(f"best: {search.best_params_}  val F1 {search.best_score_:.4f}")

    if val_only:  # kept apart from outputs/results/*.json, which hold only test-evaluated models
        (RESULTS_DIR / "validation" / name).parent.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "validation" / f"{name}.json").write_text(json.dumps(validation, indent=2))
        return

    (MODELS_DIR / name).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, MODELS_DIR / f"{name}.joblib")
    X_test = make_input(recs, sp["test"])
    p_test = np.concatenate([search.predict(X_test[i:i + CHUNK]) for i in range(0, len(X_test), CHUNK)])
    report(name, labels(recs, sp["test"]), p_test, validation)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    stride = [int(a.split("=")[1]) for a in sys.argv if a.startswith("--stride=")]
    main(*args, val_only="--val-only" in sys.argv, stride=stride[0] if stride else 10,
         protocol_b="--protocol-b" in sys.argv)
