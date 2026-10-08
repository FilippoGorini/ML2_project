"""Model inputs: turn window ends into a feature matrix (one row per window).

    none      no input at all (baseline)
    current   the 54 channels at the window's last timestep (no history)
    window_*  raw past timesteps at the lags (ms before the label) below, oldest first;
              15 lags x 54 channels = 810 values, except window_full (150 x 54 = 8,100)
    feat_g1   hand-crafted, linear summaries of the window (216): current value, change over the last
              5 ms and 50 ms, window mean — per channel
    feat_g2   hand-crafted, non-linear (174): window std, min, max per channel; per foot: height above its
              window minimum, speed now, mean speed over the last 20 ms
    feat_all  feat_g1 + feat_g2 (390)
    feat_g1_stats / feat_g1_foot   feat_g1 + only one half of feat_g2 (ablation): std/min/max (162) or the
              12 foot features

The 150 ms window is the dataset paper's convention (Lin et al. 2021); which lags to keep inside it is
chosen on validation. Joint/foot data are 500 Hz upsampled to 1 kHz, so consecutive timesteps often repeat.
"""
import numpy as np

from src.data import WINDOW

LAGS = {
    "window_uniform": np.arange(0, WINDOW, 10),                       # every 10 ms over 150 ms
    "window_log": np.array([0, 1, 2, 3, 5, 8, 12, 18, 26, 37, 52, 72, 98, 128, 149]),  # dense recent, sparse old
    "window_last15": np.arange(15),                                   # last 15 ms, every 1 ms
    "window_last30": np.arange(0, 30, 2),                             # last 30 ms, every 2 ms
    "window_full": np.arange(WINDOW),                                 # all 150 ms (ceiling, 8,100 values)
}


def none(recs, ends):
    return np.zeros((sum(len(e) for e in ends.values()), 1), dtype=np.float32)


def current(recs, ends):
    return np.concatenate([recs[name][0][e] for name, e in ends.items()])


def lagged(lags):
    lags = np.sort(lags)[::-1]  # oldest first, the current timestep (lag 0) last
    return lambda recs, ends: np.concatenate(
        [recs[name][0][e[:, None] - lags].reshape(len(e), -1) for name, e in ends.items()])


P_Z = [30 + 3 * k + 2 for k in range(4)]            # foot height (z of p) of RF, LF, RH, LH
V_FOOT = [slice(42 + 3 * k, 45 + 3 * k) for k in range(4)]  # foot velocity (x, y, z) of each leg


def _g1(X, e, W):
    return [X[e], X[e] - X[e - 5], X[e] - X[e - 50], W.mean(2)]


def _g2_stats(X, e, W):
    return [W.std(2), W.min(2), W.max(2)]


def _g2_foot(X, e, W):
    speed = np.stack([np.linalg.norm(X[:, v], axis=1) for v in V_FOOT], 1)  # (N, 4) foot speed
    return [X[e][:, P_Z] - W[:, P_Z].min(2),              # 0 when the foot is at its lowest point
            speed[e],
            speed[e[:, None] - np.arange(20)].mean(1)]


def _g2(X, e, W):
    return _g2_stats(X, e, W) + _g2_foot(X, e, W)


def handcrafted(*groups, chunk=5000):
    def make(recs, ends):
        out = []
        for name, e_all in ends.items():
            X = recs[name][0]
            windows = np.lib.stride_tricks.sliding_window_view(X, WINDOW, axis=0)  # (N-149, 54, 150), a view
            for i in range(0, len(e_all), chunk):     # chunks: a window block is 54 x 150 values per row
                e = e_all[i:i + chunk]
                W = windows[e - WINDOW + 1]           # the 150 steps ending at e
                out.append(np.concatenate([f for g in groups for f in g(X, e, W)], 1).astype(np.float32))
        return np.concatenate(out)
    return make


INPUTS = {"none": none, "current": current, **{name: lagged(lags) for name, lags in LAGS.items()},
          "feat_g1": handcrafted(_g1), "feat_g2": handcrafted(_g2), "feat_all": handcrafted(_g1, _g2),
          "feat_g1_stats": handcrafted(_g1, _g2_stats), "feat_g1_foot": handcrafted(_g1, _g2_foot)}
