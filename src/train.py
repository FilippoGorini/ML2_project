"""Train one model on the default split and report test scores.

    python -m src.train majority
"""
import sys

import numpy as np
from sklearn.dummy import DummyClassifier

from src.data import labels, load_recordings, split
from src.evaluate import report

MODELS = {
    # ignores the input, always predicts each leg's most frequent training label
    "majority": lambda: DummyClassifier(strategy="most_frequent"),
}


def main(name):
    recs = load_recordings()
    sp = split(recs)
    y_train, y_test = labels(recs, sp["train"]), labels(recs, sp["test"])

    model = MODELS[name]().fit(np.zeros((len(y_train), 1)), y_train)
    report(name, y_test, model.predict(np.zeros((len(y_test), 1))))


if __name__ == "__main__":
    main(sys.argv[1])
