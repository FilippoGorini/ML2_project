"""Load the Mini-Cheetah recordings and split them (protocol A: within each recording).

A sample is a window of the last WINDOW timesteps (54 channels each); its label is
the 4-leg contact state at the window's LAST timestep. Windows are referenced by
their end index, so stride 1 = one sample per timestep.

Split, per recording, in time order:   [ train 70% | gap | val 15% | gap | test 15% ]
The gaps are WINDOW steps long, so no timestep is shared between two sets.
"""
from pathlib import Path

import numpy as np
from scipy.io import loadmat

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mini_cheetah_contact_datasets"
FIELDS = ["q", "qd", "imu_acc", "imu_omega", "p", "v"]  # 12+12+3+3+12+12 = 54 channels
LEGS = ["RF", "LF", "RH", "LH"]
WINDOW = 150  # timesteps; data is sampled at 1 kHz -> 150 ms
TRIM_END = 300  # timesteps dropped at the end of every recording: there the labels stop
                # (all feet "in the air" for up to ~270 ms) while the robot is still walking


def load_recordings(data_dir=DATA_DIR):
    """{name: (X (N, 54) float32, y (N, 4) uint8)}, one entry per .mat file."""
    recs = {}
    for path in sorted(Path(data_dir).rglob("*.mat")):
        m = loadmat(path)
        X = np.concatenate([m[f] for f in FIELDS], axis=1).astype(np.float32)[:-TRIM_END]
        y = m["contacts"].astype(np.uint8)[:-TRIM_END]
        recs[path.stem] = (X, y)
    return recs


def split_ends(n, train=0.7, val=0.15, stride=10, test_stride=1):
    """Window end indices (train, val, test) for one recording of n timesteps.
    Train and val use `stride` (val is scored many times in grid searches); test uses
    `test_stride` (scored once per model, so every timestep can be scored)."""
    first = WINDOW - 1  # first index with a full window behind it
    a = first + int((n - first) * train)
    b = first + int((n - first) * (train + val))
    return (np.arange(first, a, stride),
            np.arange(a + WINDOW, b, stride),  # first val window starts right after the last train step
            np.arange(b + WINDOW, n, test_stride))


def split(recs, stride=10, test_stride=1):
    """{"train"|"val"|"test": {name: end indices}} for every recording."""
    out = {"train": {}, "val": {}, "test": {}}
    for name, (X, _) in recs.items():
        for part, ends in zip(out, split_ends(len(X), stride=stride, test_stride=test_stride)):
            out[part][name] = ends
    return out


def labels(recs, ends):
    """(M, 4) contact labels at the given window ends, recordings stacked in order."""
    return np.concatenate([recs[name][1][e] for name, e in ends.items()])


def check_no_overlap(sp):
    """Assert that, in every recording, train/val/test windows share no timestep."""
    for name in sp["train"]:
        tr, va, te = (sp[p][name] for p in ("train", "val", "test"))
        # a window ending at e covers timesteps [e - WINDOW + 1, e]
        assert tr.max() < va.min() - WINDOW + 1, f"train/val overlap in {name}"
        assert va.max() < te.min() - WINDOW + 1, f"val/test overlap in {name}"


if __name__ == "__main__":
    recs = load_recordings()
    sp = split(recs)
    check_no_overlap(sp)
    print(f"{len(recs)} recordings, no train/val/test overlap (train/val stride 10, test stride 1)\n")
    print(f"{'recording':30s} {'steps':>7s} {'train':>7s} {'val':>7s} {'test':>7s}")
    for name, (X, _) in recs.items():
        print(f"{name:30s} {len(X):7d} " + " ".join(f"{len(sp[p][name]):7d}" for p in sp))
    print(f"{'total':30s} {sum(len(X) for X, _ in recs.values()):7d} "
          + " ".join(f"{sum(len(e) for e in sp[p].values()):7d}" for p in sp))
