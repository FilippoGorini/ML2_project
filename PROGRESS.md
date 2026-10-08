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

## Phase 3 — Baseline and first model *(next)*
Plan: majority baseline; ridge on the current timestep only (54 values, no history).

## Planned
- Ridge on the raw window → does history help?
- Hand-crafted features (incl. `|v|`, `v²`, window statistics) → does domain knowledge help?
- Kernel ridge (RBF) and SVM on the best input → boundary vs loss (EMG-style 2×2).
- Possibly a tree model (feature importance is easy to present).
- Appendix: stride 1 vs 150 rerun; class weighting / threshold for precision; random-split comparison
  to show the optimism we avoid.

## Presentation one-liners
- "We predict whether each foot is on the ground now, from the last 150 ms of proprioception."
- "Consecutive windows are correlated, so the data aren't i.i.d.; inside the training set that only
  reduces the effective sample size. What matters is independence between training and test, so we
  split each recording into contiguous blocks with gaps."
- "We dropped the last 300 ms of each recording because the labels stop before the sensor data."
- "False contacts are worse than missed contacts for the EKF, so we report precision separately."
