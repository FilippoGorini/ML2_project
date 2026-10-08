# Project progress — foot-contact estimation (Mini-Cheetah)

Running log of decisions, reasons and results, in the order they happened.
Written to be turned into the presentation: each phase = one or two slides.

---

## Phase 0 — Restart (Oct 8, 2026)

The first iteration (many models, many splits) was archived on the `old` branch
(`old_REPORT.md` there has every number). Restarted from scratch to build a few things,
one at a time, each one understood and justified.

**Course constraint:** deep learning is not a mandatory section for us. The core is
classical models (ridge, kernel ridge, SVM, trees). Neural nets only as a possible appendix.

**Lessons from colleagues' projects** (EMG gesture recognition, wall-following robot, animal images):
- Best structure (EMG): change **one thing at a time** — e.g. ridge → kernel ridge changes only the
  boundary, ridge → linear SVM changes only the loss — so every result has a single cause.
- All three split time-ordered data at random → test samples sit next to training samples → optimistic
  scores. We do better (Phase 2).
- Other pitfalls to avoid: no fixed seed, tuning hyperparameters under a different objective than the
  final model (class weights added after grid search), different CV procedures per model, no baseline.

---

## Phase 1 — Data

**Dataset:** MIT Mini-Cheetah contact dataset (UMich CURLY, Lin et al. 2021).
15 recordings, 1 kHz, ~1M timesteps (~17 min). 8 terrains, 3 gaits (trot, pronk, gallop) + 2 recordings
with the robot held in the air (legs moving, never in contact).

**Input:** 54 proprioceptive channels per timestep: joint angles `q` (12), joint velocities `qd` (12),
IMU acceleration (3) and angular velocity (3), foot positions `p` (12) and velocities `v` (12) in the body frame.

**Sample:** a window of the last 150 timesteps (150 ms ≈ half a stride).
**Label:** contact state of the 4 feet (`RF, LF, RH, LH`) at the window's **last** timestep →
"from the last 150 ms, is each foot on the ground *now*?" This is exactly how it would run on the robot.

**Application:** the contact estimate feeds a contact-aided EKF for state estimation. The EKF treats a
foot in contact as fixed on the ground. Hence:
- **false contact** (foot swinging, model says contact) → wrong velocity injected → estimate drifts. **Harmful.**
- **missed contact** → the EKF skips one measurement. **Mild.**

### Exploration findings (`notebooks/01_explore.ipynb`)
- **No stand-up / sit-down transients**: body height constant, regular gait from the first sample.
  (An apparent dip at the edges was a plotting artefact — zero-padded moving average.)
- **Label artefact at the end**: in the last 100–270 ms of every ground recording the labels stop
  (all feet "in the air") while the robot is still walking. Would have landed entirely in the test set.
  → **drop the last 300 ms of every recording** (`TRIM_END` in `src/data.py`).
  Rule for trimming: only when labels are wrong, not when data "looks different" — realistic
  but unusual data stays (it is what the deployed model will face).
- **Air recordings kept**: legs move, nothing touches → exactly the examples that teach "no false contacts".
- **Class balance**: each foot in contact ~35% of the time on the ground (pronk/gallop more), ~32% overall.
- **Leg coupling**: 4 of the 16 joint states cover ~95% of timesteps (`0000` 37%, `1001` 26%, `0110` 26%,
  `1111` 6%) — trot diagonal pairs.
- **Two body-height groups**: `concrete_*` + `grass` at ~0.34 m, other terrains at ~0.26 m.
  → foot height alone cannot be thresholded globally (contact in one group overlaps swing in the other).
- **Not linearly separable in raw signals**: in contact the foot velocity is ≈ 0; in swing it is
  positive *and* negative. "Close to zero" is not a linear function of `v_z` (same issue as the EMG
  amplitude). A linear model needs `|v_z|`, `v_z²` (features) or a kernel. → motivates Phases 4–5.
- **Autocorrelation**: signals oscillate with the gait (period ≈ 265 ms); at 150 ms lag ≈ −0.5,
  three strides later still ≈ 0.8. Consecutive samples are far from independent.

---

## Phase 2 — Evaluation protocol

### Split: protocol A, contiguous blocks inside each recording
```
each recording:  [ train 70% | gap | val 15% | gap | test 15% ]      gap = 150 steps (one window)
```
- Test comes from the same distribution (same recordings) as train/val — the course's protocol A.
- **Contiguous blocks, not random windows.** A random split puts test windows right next to training
  windows (thousands of boundaries) → the model interpolates between near-identical samples → optimistic
  score. Blocks have 2 boundaries per recording, and the gaps guarantee no timestep is shared
  (`check_no_overlap` asserts it).
- Is the end of a recording representative? Checked: contact ratio of train/val/test blocks agrees
  within ~0.03 for every recording → plain blocks are fine (no need for interleaved chunks).

### Overlapping windows and the i.i.d. question
Consecutive samples are correlated, so the data are not i.i.d. — true for *any* sampling of a time
series, not only overlapping windows (non-overlapping windows 150 ms apart are still correlated ≈ −0.5).
Where it matters and how it is handled:

| Where | Risk | Handled by |
|---|---|---|
| train ↔ test | optimistic test score | contiguous blocks + gaps |
| train ↔ val (tuning) | choosing hyperparameters that memorise | same: val is its own block, no random k-fold |
| within train | fewer *effective* samples than windows; no bias | nothing needed; stride = compute knob |
| error bars | intervals computed with n = #windows far too narrow | don't; vary across recordings instead |

Honest phrasing: "~700k training windows from ~12 minutes of walking (≈ 2,400 gait cycles on the ground)", not "700k samples".

**Experiment** (ridge and gradient boosting, train stride 1 / 10 / 150, always tested on every timestep;
scratch code, run before the 300 ms trim — conclusions unaffected):

| Model | stride 1 | stride 10 | stride 150 (non-overlapping) |
|---|---|---|---|
| ridge, current timestep (54) | 0.8325 | 0.8326 | 0.8308 (±0.001 across offsets) |
| ridge, raw window (15×54) | 0.9178 | 0.9174 | 0.9052 |
| boosting, current timestep | 0.9493 | 0.9493 | 0.935 |

(test macro per-leg F1)
- stride 1 = stride 10: 90% of overlapping windows are redundant, but harmless.
- Non-overlapping windows only **lose** information (−1.2 to −1.4 points for the bigger models).
- They make over-fitting **worse**: boosting train F1 0.983 → 0.995 → 1.000, train–test gap 3.4 → 6.5 points.
- Results vary more with which windows are kept (thinning adds noise).

**Decision:** overlapping windows, **training stride 10** as the default configuration (same result as
stride 1, 10× cheaper). Re-running with stride 1 and 150 → possible appendix.
**Val stride 10, test stride 1** (`split(recs)` defaults):
- val is scored once per hyperparameter combination in every grid search → stride 10 keeps tuning fast;
  neighbouring timesteps are nearly identical, so the estimate barely changes.
- test is scored once per model → every timestep of the test blocks is scored, as on the robot.

Windows: 70,476 train / 14,882 val / 148,763 test.

### Problem formulation: 4 binary problems (one per leg)
- Each leg is a binary classifier (contact vs no contact); **each sees all 54 channels** (other legs'
  signals included, so leg coupling is still usable).
- Why not one 16-class model (what the original paper did): ridge / SVM / kernel ridge are natively
  binary; classes ~32/68 instead of 16 very imbalanced classes; the EKF consumes per-leg contacts anyway.
- For ridge it is one linear solve with 4 output columns.

### No class weighting (for now)
- Imbalance is mild (32/68), and the 4 legs have the same rate → one weighting would serve all.
- Up-weighting contact makes the model say "contact" more → more **false contacts**, the harmful error.
- Weighting mostly shifts the decision threshold. If wanted later: one hyperparameter chosen on val.

### Metrics (contact = positive class)
- **Main: macro per-leg F1** (F1 per leg, averaged). Ignores true negatives, so "always no contact" can't score.
- **Precision and recall** separately — precision protects the EKF.
- **Exact-state accuracy** (all 4 legs right at once) — comparable with the 16-class papers.
- Accuracy only for reference: "always no contact" already gets ~68%.

### Baseline
**Majority class** ("all feet in the air", `DummyClassifier`): accuracy ~68%, contact F1 = 0.
Gives every later number a zero point and catches models that collapse to "never contact".

---

## Phase 3 — Baseline and first model

Code: `src/train.py` (fit + report), `src/evaluate.py` (metrics, saves `outputs/results/<model>.json`).
Every model reports test scores on every timestep (headline) and on every 10th (check that the
stride-10 validation is a faithful proxy).

### Majority baseline (`python -m src.train majority`)
`DummyClassifier(strategy="most_frequent")` on the 4 label columns = 4 independent "most frequent label"
rules, one per leg. Every leg's most frequent label is "no contact" → always predicts all feet in the air.

| | F1 | precision | recall | accuracy | exact-state |
|---|---|---|---|---|---|
| majority (test, every timestep) | 0.000 | 0.000 | 0.000 | 0.665 | 0.368 |

- Accuracy 0.665 without looking at the input → accuracy alone is a misleading metric here.
- Exact-state 0.368 = share of test timesteps with all feet in the air (air recordings + trot flight phases).
- Every 10th timestep: exact-state 0.3685 vs 0.3679 → the subsample agrees.

### Next
Ridge on the current timestep only (54 values, no history).

## Plan

### Inputs (each adds one thing)
| Input | Size | Question |
|---|---|---|
| **A. current timestep** | 54 | how much does one instant tell? |
| **B. raw window** | 150 × 54; every 10 ms → 15 × 54 = 810 (full 8,100 ≈ 4.5 GB at stride 10) | does history help? |
| **C. hand-crafted features** from the window | a few hundred (per-channel stats, last values, deltas, `|v|`, `v²`, …) | does domain knowledge beat raw history? |

### Two axes, one change at a time
1. **Input ladder**, model fixed (ridge): A → B → C.
2. **Model ladder**, input fixed (best of A/B/C): ridge → kernel ridge (boundary) → SVM (loss) → trees.

### Course models: tier list for this project
✓✓ very useful · ✓ useful · ~ possible but limited · ✗ not sensible. Sections refer to `Lessons/ML2_Topics_Index.md`.

| Tier | Model (§) | A | B | C | Why |
|---|---|---|---|---|---|
| **S** | Ridge / RLS (§3) | ✓✓ | ✓✓ | ✓✓ | closed form, seconds; fixed model of the input ladder; the linear floor |
| **A** | Kernel ridge, RBF (§4) | ✓✓ | ~ | ✓✓ | main non-linear course model, should fix "`v_z` near zero"; n³ cost → train on 10–20k subsample; distances on 810 raw values less meaningful |
| **A** | Gradient boosting (§8) | ✓✓ | ✓ | ✓✓ | strongest on tabular data (0.949 on A alone in the stride experiment); fast, no scaling |
| **B+** | Random forest / bagging (§8) | ✓ | ~ | ✓✓ | non-linear, little tuning; feature importance is a good slide |
| **B+** | Kernel SVM, RBF (§7) | ✓ | ~ | ✓ | same boundary as kernel ridge, hinge loss → EMG-style 2×2; slow on 70k × 4 legs → subsample |
| **B** | Linear SVM (§7) | ✓ | ✓ | ✓ | isolates the loss (ridge vs hinge, same boundary); one controlled comparison |
| **B** | LASSO / L1-SVM (§3, §7) | ~ | ✓✓ | ✓ | on raw windows the zero weights show which lags/channels matter; interpretation more than score |
| **B−** | MLP (§10) | ✓ | ✓ | ✓✓ | "φ fixed (kernel) vs φ learned"; optional (DL not mandatory); 0.930 on features in the old runs |
| **C** | Decision tree (§8) | ✓ | ✗ | ✓ | high variance; useful only as a picture (depth-3 tree = learned thresholds) |
| **C** | Bayesian linear regression (§9) | ✓ | ✓ | ✓ | same predictions as ridge (MAP); its predictive variance is interesting for the EKF — a remark, not a model |
| **D** | 1D CNN / Transformer (§12–13) | ✗ | ✓ | ✗ | raw windows only; appendix curiosity; lost to features + MLP in the old runs |
| **D** | Perceptron (§10) | ~ | ~ | ~ | historical; ridge does linear classification better |
| — | SVR (§7) | ✗ | ✗ | ✗ | regression method |

Not in the course (left out): kNN, logistic regression, RNN/LSTM.

Course link for error bars: §9 covers the binomial estimator and the Clopper-Pearson interval — the
frequentist confidence interval on a test error rate. It assumes independent test samples, so with our
correlated windows it would be far too narrow (see Phase 2).

### Core path
1. ✅ Majority baseline.
2. Ridge on A → B → C (input ladder).
3. Kernel ridge on the best input (non-linear boundary).
4. Linear SVM (+ RBF SVM if time) on the same input (loss) → 2×2 with ridge / kernel ridge.
5. Gradient boosting or random forest on the same input, with feature importance.

### Appendix candidates
LASSO on the raw window (which lags matter); MLP; stride 1 vs 150 rerun; random-split comparison
(the optimism we avoid); class weighting / threshold for precision.

## Presentation one-liners
- "We predict whether each foot is on the ground now, from the last 150 ms of proprioception."
- "Consecutive windows are correlated, so the data aren't i.i.d.; inside the training set that only
  reduces the effective sample size. What matters is independence between training and test, so we
  split each recording into contiguous blocks with gaps."
- "We dropped the last 300 ms of each recording because the labels stop before the sensor data."
- "False contacts are worse than missed contacts for the EKF, so we report precision separately."
