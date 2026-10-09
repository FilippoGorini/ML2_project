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

**Where 150 comes from:** a convention. The dataset paper (Lin et al., CoRL 2021, arXiv 2106.15713)
uses w = 150 without justification or ablation ("to allow the network to infer from the time domain");
later papers on this dataset (Ordonez-Apraez et al. 2023, MI-HGNN 2024) keep it "for a fair comparison".
No window-length ablation found on this dataset. → we keep 150 ms as the span and choose which lags
inside it to use on validation (Phase 3).

**How the labels were made** (Lin et al. 2021, Sec. 4.3 + Appendix): not force sensors — computed
**offline from the foot height** in the hip frame: low-pass filter, local minima/maxima using past **and
future** samples, contact = between the minima (cut-off differs for trot vs pronk/gallop). Consequences:
- the label is a smoothed, non-causal function of one of our inputs → foot height dominates simple models;
- the model must approximate, from the past only, a filter that also looks ahead → history helps;
- labels are themselves an estimate: some "errors" at touchdown/lift-off may be label noise.

**Sampling:** joint/foot data recorded at 500 Hz and upsampled to 1 kHz (IMU at 1 kHz). In our files
the value repeats on ~63% of consecutive timesteps for joints/feet and ~79% for the IMU (effective update
every ~3–5 ms) → a 150-step window holds only ~40–50 distinct readings per channel.

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
  Root cause (from the label algorithm above): it needs a *future* peak to close the last contact, so
  the last stride of each recording never gets labelled.
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

### Ridge — current timestep (`python -m src.train ridge current`)
Standardized inputs → `RidgeClassifier`, α ∈ 10⁻³…10⁵ on validation, refit on train+val.

| | F1 | precision | recall | accuracy | exact-state |
|---|---|---|---|---|---|
| ridge, current (test) | 0.840 | 0.872 | 0.811 | 0.897 | 0.799 |

- Validation curve flat up to α = 10 → **under-fitting**: 54 weights, 85k samples; the limit is the
  linear boundary, not the tuning (as in the EMG project).
- Precision > recall: errors are mostly missed contacts (safe side for the EKF).
- Weak on pronking (0.61) and galloping (0.58); trot terrains 0.85–0.95. Val (0.888) > test (0.840):
  gap concentrated in a few recordings (gait drift between middle and end of a recording).
- Weights: each leg uses its own foot height and knee angle, and its **trot partner's** foot height;
  foot velocities barely used (not linearly usable).

### Window design (validation only, ridge)
Which lags inside the 150 ms window? All variants 15 lags × 54 = 810 values (same size, fair comparison);
chosen on **validation**, test run only for the winner. (Probably not presented — the presentation can
start from the chosen window.)

| input | lags (ms before label) | val F1 |
|---|---|---|
| current | 0 | 0.8875 |
| last 15 ms | 0…14 | 0.9182 |
| last 30 ms | 0, 2, …, 28 | 0.9288 |
| uniform | 0, 10, …, 140 | 0.9412 |
| **log-spaced** | 0, 1, 2, 3, 5, 8, 12, 18, 26, 37, 52, 72, 98, 128, 149 | **0.9459** |

- **Longer context matters**: 15 → 30 → 150 ms keeps improving (gait phase is invisible in short windows).
- **Recent detail matters**: at equal size and span, dense-recent (log) beats uniform by 0.5 points —
  plain 10 ms subsampling loses the last milliseconds, where touchdown shows up first.
- Full 150 × 54 window (8,100 values) not run (needs ~15–20 GB); possible ceiling check on the workstation.
- α again barely matters; "ill-conditioned" warnings expected (repeated samples → identical columns).

→ **chosen raw-window input: `window_log`.**

### Ridge — raw window, log-spaced lags (`python -m src.train ridge window_log`)

| | F1 | precision | recall | accuracy | exact-state |
|---|---|---|---|---|---|
| ridge, current (test) | 0.840 | 0.872 | 0.811 | 0.897 | 0.799 |
| **ridge, window_log (test)** | **0.930** | 0.934 | 0.925 | 0.953 | 0.879 |

- **History is worth +9 points** for a linear model (best α = 0.1, curve flat again).
- Gains exactly where the current timestep failed: **pronking 0.61 → 0.91, galloping 0.58 → 0.78**;
  other terrains +0.01…0.08; false contacts in the air recordings 1.4% → 0.
- Val → test gap shrinks from 4.7 to 1.6 points (less sensitive to gait drift within a recording).
- Precision and recall now balanced (0.93 / 0.93). LF still the hardest leg (0.914).
- Weight per lag: lag 0 largest, but all lags up to 149 ms used; opposite-sign weights on near-identical
  recent lags = differences → history lets a linear model compute derivatives/trends.

### Hand-crafted features (input C)
Design principle: for ridge, features that are **linear** in the raw window (means, deltas) add little —
ridge can already form any weighted sum of lags. Features help a linear model only if **non-linear**
(spread, extremes, absolute values). Linear summaries still matter for compactness (kernels, trees).

All computed over the 150 ms window ending at the label (`src/features.py`):

| group | features | values | reason |
|---|---|---|---|
| **g1** linear summaries | current value; change over last 5 ms and 50 ms; window mean — per channel | 216 | strongest single input; touchdown = abrupt stop (5 ms); descending/rising over the stride (50 ms); low-pass context |
| **g2 stats** | window std, min, max — per channel | 162 | activity level (EMG lesson: amplitude); extremes of the stride |
| **g2 foot** | per foot: height above its window minimum; speed ‖v‖ now; mean speed over last 20 ms | 12 | "foot at its lowest" = the label's definition, and removes the body-height offset; "foot stationary" — the non-linear fix for `v_z` near zero |

Validation F1 (ridge, best α; α flat again everywhere):

| input | values | val F1 |
|---|---|---|
| window_log (raw) | 810 | 0.9459 |
| feat_g1 | 216 | 0.9351 |
| feat_g2 | 174 | 0.9341 |
| **feat_all** (g1 + g2) | 390 | **0.9513** |

- Each group alone ≈ 1 point **below** the raw window; g1 lacks the fine recent lags (only a 5 ms delta).
- The non-linear group alone carries as much as the linear one.
- Together **+1.6 over either alone, +0.5 over the raw window with half the values** → complementary.

Ablation of g2 (validation):

| input | values | val F1 | gain over g1 |
|---|---|---|---|
| feat_g1 | 216 | 0.9351 | — |
| + foot | 228 | 0.9407 | +0.6 |
| + stats | 378 | 0.9491 | +1.4 |
| + both (feat_all) | 390 | **0.9513** | +1.6 |

- Generic stats give most of the gain; the 12 foot features are ~5× more efficient **per value**.
- On top of stats the foot features add only +0.2: "height above window minimum" = current − min, which
  ridge can already form from g1 (current) + stats (min) → redundant there; the rest comes from foot speed ‖v‖.
→ **chosen: `feat_all`** (best on val; foot features cost 12 values and are the most explainable).

All these choices (window lags, feature groups) were made with `--val-only`: each option fitted on train
and scored on validation; test inputs never built. Test is used once, for the chosen input.

### Ridge — hand-crafted features, test (`python -m src.train ridge feat_all`)

| input (ridge) | values | F1 | precision | recall | accuracy | exact-state | false-contact rate | missed-contact rate |
|---|---|---|---|---|---|---|---|---|
| current | 54 | 0.840 | 0.872 | 0.811 | 0.897 | 0.799 | 6.0% | 18.9% |
| window_log | 810 | 0.930 | 0.934 | 0.925 | 0.953 | 0.879 | 3.3% | 7.5% |
| **feat_all** | 390 | **0.946** | 0.938 | 0.955 | 0.964 | 0.899 | 3.2% | 4.6% |

(rates over all legs: false contacts / true air timesteps, missed contacts / true contact timesteps)

- **Features beat the raw window by +1.7 points with half the values** — domain knowledge pays off even for
  a linear model, because the useful features are non-linear in the raw signals.
- **The whole gain is found contacts**: missed contacts 7.5% → 4.6%, false contacts unchanged (3.3% → 3.2%).
  Recall now exceeds precision only because recall rose — the model did not become riskier for the EKF.
- **Biggest gains on the non-trot gaits again**: pronking 0.906 → 0.958, galloping 0.781 → 0.832, and grass
  0.916 → 0.949; trot terrains already ~0.95–0.97 change by ±0.01.
- **Val → test gap only 0.5 points** (0.951 → 0.946; was 4.7 for current, 1.6 for the window): window
  statistics are less sensitive to the gait drift within a recording.
- α flat again → still a **linear model limited by its boundary**; the input ladder is done:
  **0 → 0.840 → 0.930 → 0.946** (baseline → instant → raw history → features).
- Galloping (0.83) and LF (0.935) remain the weak spots.
- Weights per value: largest on the current value and on **foot height above its window minimum** (the
  feature mirroring how labels were made); small on 5 ms change and foot speed. Correlated features share
  weight, so the validation ablation, not the weights, is the measure of importance.

→ **`feat_all` is the input for the model ladder** (kernel ridge, SVM, trees).

---

## Phase 4 — Model ladder (input fixed: `feat_all`)

### Kernel methods are trained on 10k windows
Kernel methods work on pairs of training samples: kernel ridge stores an n × n kernel matrix and solves it
in ~n³ operations. At the full training set (70,476 windows; 85,358 for the train+val refit) the matrix is
**40–58 GB** in float64 → impossible here (14 GB) and on the lab workstation (30 GB). The problem is **RAM**;
the time (~1–2 h for the grid on the workstation) would be acceptable. (The EMG group had ~7k training
windows → 0.4 GB, no issue.)
- Kernel methods use **10,000 training windows taken evenly in time** (`Subsampled`, `N_KERNEL` in
  `src/models.py`) — the same windows for every kernel model (kernel ridge now, RBF SVM later).
- Only kernel methods need this; ridge, linear SVM, trees keep the full stride-10 training set.
- **Fair comparison**: ridge is also trained on the same 10k windows (`ridge_10k`): ridge vs ridge_10k = cost of
  the subsample; ridge_10k vs kernel ridge = effect of the non-linear boundary on identical data.
- Test is always every timestep.

### Kernel ridge, RBF (`python -m src.train krr feat_all`)
Same loss and penalty as ridge, RBF kernel k(x, x′) = exp(−γ‖x − x′‖²), targets ±1, contact if output > 0.
sklearn's `KernelRidge` has no bias term → **the mean target is subtracted before fitting and added back at
prediction** (= the intercept ridge has; without it the 32/68 offset would have to be built from the kernel).
γ = s / 390. Grid α ∈ {0.01, 0.03, 0.1, 0.3, 1} × s ∈ {0.1, 0.2, 0.3, 0.5, 1, 2, 3} (35 combinations, ~15 min on this PC).

| model (feat_all) | train windows | val F1 | test F1 | precision | recall | exact-state | false contact | missed contact |
|---|---|---|---|---|---|---|---|---|
| ridge | 70k | 0.951 | 0.946 | 0.938 | 0.955 | 0.899 | 3.2% | 4.6% |
| ridge_10k | 10k | 0.951 | 0.942 | 0.936 | 0.949 | 0.891 | 3.3% | 5.1% |
| **kernel ridge (RBF)** | 10k | 0.963 | **0.956** | 0.951 | 0.960 | 0.909 | 2.5% | 4.0% |

- **The subsample costs ridge little**: same validation, −0.4 on test.
- **The RBF kernel adds +1.4 on identical data** (+1.0 over ridge on 7× more windows): a non-linear boundary
  helps on top of the features.
- **Both errors drop**: false contacts 3.3% → 2.5%, missed 5.1% → 4.0%.
- **Gains on every ground recording**, largest on the hardest: galloping 0.818 → 0.868; pronking, small
  pebble, old asphalt ≈ +0.02; grass, rock road ≈ 0.
- **Hyperparameters: flat optimum** along a diagonal band (smaller α pairs with smaller γ); best α = 0.03,
  s = 0.3 (val 0.9633), inside the grid; ~10 combinations within 0.001 of it → **performance does not hinge on
  tuning**. On a flat plateau the validation winner is partly chance ("winner's curse"), so validation differences
  below ~0.1 point are not real.
- Large γ (s ≥ 2) clearly worse at every α: kernel too local for 10k windows in 390-D. α and γ interact → searched together.
- Notebook: `VAL_PLOT_3D` at the top switches the validation plot between a 2-D heatmap and a 3-D surface.

### SVMs: boundary × loss
Every model computes a score f(x) per leg, contact if f(x) > 0. Two independent choices:
- **boundary**: hyperplane (ridge, linear SVM) or RBF kernel (kernel ridge, RBF SVM);
- **loss** (margin m = y·f(x)): **squared** (1 − m)² pulls every window to ±1 (also penalizes "too correct" ones);
  **hinge** max(0, 1 − m) is zero beyond the margin → only windows near the boundary (support vectors) matter.
  SVM objective ½‖w‖² + C·Σ hinge: C weighs the loss → opposite role of α (roughly C ≈ 1/(2α)).

Models (all on `feat_all`, standardized, one binary SVM per leg, no class weights, same selection/refit/test):
- `svm_linear`: `LinearSVC(loss="hinge")`, 70k windows (like ridge), C ∈ {0.001 … 1}.
- `svm_linear_10k`: the same on the 10k kernel windows (hinge counterpart of `ridge_10k`).
- `svm_rbf`: `SVC(kernel="rbf")`, the same 10k windows as kernel ridge, γ = s/390,
  C ∈ {0.1 … 100} × s ∈ {0.1, 0.3, 1, 3}.
Run on the lab workstation (`max_iter` = 1M for the linear SVM: large C converges slowly — the final 70k fits
needed 139k–264k iterations, all converged; total ~5 h, the RBF SVM ~15 min).

**2 × 2 on identical data (10k windows), test F1:**

| | squared loss | hinge loss | effect of the loss |
|---|---|---|---|
| hyperplane | ridge_10k 0.942 | svm_linear_10k 0.942 | 0.000 |
| RBF kernel | **kernel ridge 0.956** | RBF SVM 0.953 | −0.003 |
| effect of the boundary | +0.014 | +0.011 | |

**All model-ladder models:**

| model (feat_all) | train windows | val F1 | test F1 | precision | recall | exact-state | false contact | missed contact |
|---|---|---|---|---|---|---|---|---|
| ridge | 70k | 0.951 | 0.946 | 0.938 | 0.955 | 0.899 | 3.2% | 4.6% |
| linear SVM | 70k | 0.959 | 0.952 | 0.946 | 0.958 | 0.903 | 2.8% | 4.2% |
| ridge_10k | 10k | 0.951 | 0.942 | 0.936 | 0.949 | 0.891 | 3.3% | 5.1% |
| linear SVM 10k | 10k | 0.953 | 0.942 | 0.934 | 0.950 | 0.883 | 3.4% | 5.0% |
| **kernel ridge** | 10k | 0.963 | **0.956** | 0.951 | 0.960 | 0.909 | 2.5% | 4.0% |
| RBF SVM | 10k | 0.962 | 0.953 | 0.947 | 0.959 | 0.903 | 2.7% | 4.1% |

Conclusions:
- **The boundary matters, the loss does not (on identical data)**: hyperplane → RBF +1.1 to +1.4; squared ↔ hinge
  0.0 to 0.3. The EMG project's conclusion, here on a fully controlled 2 × 2.
- **But with all the data the hinge wins for the hyperplane**: linear SVM 70k 0.952 vs ridge 0.946. 10k → 70k gives
  the SVM +1.0, ridge only +0.4: the hinge is decided by the hard windows near the boundary (touchdown/lift-off),
  and 7× more data means 7× more of them; the squared loss is dominated by the easy bulk.
- **Best model: kernel ridge (0.956)** — lowest false-contact rate (2.5%), best LF (0.950). The 70k linear SVM comes
  within 0.4 points (level with the RBF SVM): the kernel models are capped at 10k by memory, so part of their
  advantage depends on that limit.
- **Per recording**: kernel models better on non-trot gaits (galloping 0.868 vs 0.846, pronking) and concrete; the 70k
  linear SVM slightly better on outdoor trot terrains (grass, forest, rock road, small pebble).
- **RBF SVM hyperparameters**: best C = 10, s = 0.3 (same s as kernel ridge), inside the grid; diagonal band again,
  flat top (0.960–0.962), s = 3 bad. C ≈ 1/(2α) with kernel ridge's α = 0.03 → C ≈ 17 ≈ chosen 10.
- **Sparsity**: RBF SVM keeps ~900–1,050 support vectors per leg (~10% of the 10k; half at the bound α = C, i.e.
  inside the margin — mostly touchdown/lift-off) → predictions ~10× cheaper than kernel ridge (relevant at 1 kHz).

### Trees: three steps
Trees replace the hyperplane / smooth kernel boundary with **axis-aligned thresholds** ("feature j < t"): a different
inductive bias, no scaling needed (invariant to monotone transforms of each feature), cost ~n log n → **all 70k
training windows**, no subsample. One change per step:

| step | model (`src/models.py`) | what changes | tuned on validation |
|---|---|---|---|
| 1 | `tree`: one decision tree per leg | thresholds instead of hyperplane / kernel | `min_samples_leaf` ∈ {1, 5, 20, 50, 100, 200} |
| 2 | `rf`: random forest, 200 trees | average many decorrelated deep trees → **lower variance** | `max_features` ∈ {√390 ≈ 20, 39, 78} × `min_samples_leaf` ∈ {1, 5, 20} |
| 3 | `gb`: gradient boosting (`HistGradientBoosting`) | add small trees one by one on the log-loss gradient → **lower bias** | `max_iter` ∈ {100…800} × `max_leaf_nodes` ∈ {15…127}, learning rate fixed 0.1 |

- **Tree**: greedy splits on one feature at a time, each leaf predicts its majority label. High variance: on near-duplicate
  windows a deep tree memorises → leaf size is the only knob. Its value: the tree → forest gap measures how much variance
  costs, and a depth-3 tree (fitted only for the picture) shows the learned thresholds.
- **Random forest**: bootstrap samples + random feature subset per split decorrelate the trees; averaging lowers variance
  without raising bias. More trees never hurt → `n_estimators` fixed, not tuned. The out-of-bag score is **not** used:
  left-out windows have near-identical neighbours in the bootstrap → optimistic (same reason as no random k-fold).
- **Gradient boosting**: F(x) = Σ shrunk small trees, each fitted to what the ensemble still gets wrong. Learning rate and
  number of trees trade off → rate fixed, trees tuned. Features binned into 255 quantile bins (fast at 70k × 390).
  Its loss is the log-loss, so vs the SVMs both boundary and loss change (the 2 × 2 showed the loss barely matters).
  `early_stopping=False`: sklearn's default (on above 10k samples) holds out a random 10% of near-duplicate training
  windows and would fit the final model on 90% only.
- **Comparable with the other models**: same `feat_all`, same 70,476 / 14,882 windows and fixed val fold, F1 macro,
  refit on train+val, test every timestep; 4 independent per-leg problems (`MultiOutputClassifier` — a multi-output
  forest would share splits across legs); no class weights; p > 0.5 ⇔ score > 0; `random_state=0`.
- **Feature importance**: grouped permutation importance on the final models (shuffle a whole group, measure the F1 drop;
  by feature type and by channel), on every 10th test timestep — post-hoc, no choice depends on it. Not the forest's
  impurity importance: biased toward many-valued features and split among correlated ones (`feat_all` is full of them).
- **Expected**: boosting best (0.949 on the 54 current values alone in Phase 2, vs ridge 0.833; uses all 70k windows),
  forest slightly below, single tree clearly below. Trees predict a constant outside the training range → may be weaker
  on rare extreme gaits (galloping).
- Compared with a colleagues' project (wall-following robot, `DecisionTreeClassifier` tuning `min_samples_leaf`, forest of
  300 tuning `max_features`): same knobs, but they tuned without class weights and retrained with them, used a random
  split + random 3-fold CV, accuracy, and impurity importance — all avoided here.

#### Results (test, every timestep)

| model (feat_all) | train windows | val F1 | test F1 | precision | recall | exact-state | false contact | missed contact |
|---|---|---|---|---|---|---|---|---|
| ridge | 70k | 0.951 | 0.946 | 0.938 | 0.955 | 0.899 | 3.2% | 4.6% |
| linear SVM | 70k | 0.959 | 0.952 | 0.946 | 0.958 | 0.903 | 2.8% | 4.2% |
| kernel ridge | 10k | 0.963 | 0.956 | 0.951 | 0.960 | 0.909 | 2.5% | 4.0% |
| decision tree | 70k | 0.951 | 0.940 | 0.938 | 0.942 | 0.876 | 3.2% | 5.9% |
| random forest | 70k | 0.972 | 0.967 | 0.965 | 0.968 | 0.928 | 1.8% | 3.2% |
| gradient boosting 10k | 10k | 0.969 | 0.962 | 0.957 | 0.967 | 0.918 | 2.2% | 3.3% |
| **gradient boosting** | 70k | 0.975 | **0.972** | 0.969 | 0.976 | 0.937 | 1.6% | 2.4% |

- **Gradient boosting is the best model by a wide margin**: +1.6 over kernel ridge, +2.0 over the 70k linear SVM;
  val agrees (0.975 vs 0.963); val → test gap 0.3, the smallest of all models.
- **Tree ladder, one change per step: 0.940 → 0.967 → 0.972.** tree → forest **+2.7** (averaging removes most of a single
  tree's variance — the largest step; the forest alone beats every non-tree model, +1.1 over kernel ridge);
  forest → boosting **+0.5** (lower bias), on both errors: false contacts 1.8% → 1.6%, missed 3.2% → 2.4%.
- Per recording: forest > kernel ridge on every ground recording, boosting > forest on every one (pronking by 0.001);
  galloping 0.868 → 0.893 → 0.914.
- **Both errors drop by more than a third** vs kernel ridge (false 2.5% → 1.6%, missed 4.0% → 2.4%); exact-state 0.937.
- **Best on every ground recording**, most on the hard ones: galloping 0.868 → 0.914, old asphalt 0.929 → 0.944,
  pronking 0.971 → 0.986. No false contacts in the air recordings.
- **One tree is not enough**: 0.940 < ridge 0.946. Ties ridge on val (0.951) but loses 1.1 on test → high variance.
  Most missed contacts (5.9%), only model with clear false contacts in the air (4.4% on air_walking_gait).
  Leaf size: 1 → 0.944 (memorises near-duplicates), 20–50 flat top 0.951, 200 → 0.940; chosen 50 (depth 28–33,
  ~230 leaves per leg).
- **Boosting hyperparameters**: whole grid 0.969–0.975 (even 100 trees × 15 leaves beats every other model on val);
  chosen 800 trees × 63 leaves (0.9750) — on the upper edge of `max_iter`, but each doubling adds less (+0.14, +0.05,
  +0.04 points at 63 leaves); 127 leaves no better. **Edge check** (val only, one train-only fit scored after every tree up to 3,200,
  scratch script): 0.9754 at 1,200 trees, then flat 0.9753 (127 leaves: flat 0.9748) → +0.04 at most, 800 stays.
- **Random forest hyperparameters**: smallest leaves best (`min_samples_leaf` = 1 at every `max_features`) — the opposite
  of the single tree (1 → 0.944 < 0.951): each deep tree memorises, averaging 200 cancels it. `max_features` barely
  matters (0.9709–0.9715); chosen 39 of 390 per split. Trees ~38 deep, ~990–1,140 leaves; 135 MB for 800 trees.
- **Depth-3 tree for RF (picture only)**: the root splits on the **LF** foot height above its window minimum (≤ 1.9 cm →
  RF in air 90%): the opposite trot pair — the tree finds the leg coupling by itself. 7 questions ≈ 94% of RF training
  windows right (majority 66%).
- **Equal data (`gb_10k`: same boosting and grid on the kernel methods' 10k windows)**: boosting vs kernel ridge changes
  boundary, loss (log-loss) and data (70k vs 10k); at equal data boosting still wins, **0.962 vs 0.956** (+0.6, val
  0.969 vs 0.963), fewer false (2.2% vs 2.5%) and missed (3.3% vs 4.0%) contacts, better or level on every ground
  recording except grass (0.947 vs 0.949). → of boosting's +1.6 over kernel ridge, **~0.6 is the model, ~1.0 the 7×
  more windows** kernel ridge cannot use (memory). 10k → 70k: boosting +1.0, like the linear SVM (ridge +0.4).
  `gb_10k` best 400 trees × 31 leaves (inside the grid, 0.965–0.969): on 10k, bigger trees no longer help.

**Grouped permutation importance** (F1 drop, every 10th test timestep, 3 shuffles; ridge for reference):

| group | ridge | tree | forest | boosting |
|---|---|---|---|---|
| channel `p` (foot positions) | 0.40 | 0.36 | 0.14 | 0.18 |
| channel `v` (foot velocities) | 0.34 | 0.16 | 0.06 | 0.09 |
| channel `q` / `qd` | 0.31 / 0.23 | 0.03 / 0.11 | 0.010 / 0.022 | 0.006 / 0.008 |
| channel `acc` / `omega` | 0.017 / 0.006 | 0.07 / 0.004 | 0.012 / 0.001 | 0.004 / 0.001 |
| type: current value | 0.23 | 0.12 | 0.044 | 0.044 |
| type: change 50 ms | 0.11 | 0.23 | 0.047 | 0.031 |
| type: foot height − min | 0.32 | 0.16 | 0.006 | 0.014 |
| type: min / max | 0.20 / 0.19 | 0.05 / 0.07 | 0.007 / 0.006 | 0.004 / 0.007 |
| type: foot speeds | ≤ 0.016 | ≤ 0.002 | ≤ 0.001 | ≤ 0.001 |

- **Foot positions, then foot velocities** matter most for every model; `omega` hardly at all.
- **The ensembles are far more robust**: no single feature type costs the forest or boosting more than 4.7 points; ridge
  loses 19–32 to several types, the tree 23 to the 50 ms change. Many trees spread over many correlated features, which stand in for each other.
- Foot speeds add ~nothing **given the rest** (std / min / max of `v` carry it). Permutation measures what a group adds
  given all others — correlated groups mask each other.

## Plan

### Inputs (each adds one thing)
| Input | Size | Question |
|---|---|---|
| **A. current timestep** | 54 | how much does one instant tell? |
| **B. raw window** | 150 × 54; every 10 ms → 15 × 54 = 810 (full 8,100 ≈ 4.5 GB at stride 10) | does history help? |
| **C. hand-crafted features** from the window | a few hundred (per-channel stats, last values, deltas, `|v|`, `v²`, …) | does domain knowledge beat raw history? |

### Two axes, one change at a time
1. **Input ladder**, model fixed (ridge): A → B → C.
2. **Model ladder**, input fixed (best of A/B/C): ridge → kernel ridge (boundary) → SVM (loss) → trees (tree → forest → boosting).

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
| **C** | Decision tree (§8) | ✓ | ✗ | ✓ | high variance; first rung of the tree ladder (tree → forest = cost of variance) and a picture (depth-3 tree = learned thresholds) |
| **C** | Bayesian linear regression (§9) | ✓ | ✓ | ✓ | same predictions as ridge (MAP); its predictive variance is interesting for the EKF — a remark, not a model |
| **D** | 1D CNN / Transformer (§12–13) | ✗ | ✓ | ✗ | raw windows only; appendix curiosity; lost to features + MLP in the old runs |
| **D** | Perceptron (§10) | ~ | ~ | ~ | historical; ridge does linear classification better |
| — | SVR (§7) | ✗ | ✗ | ✗ | regression method |

Not in the course (left out): kNN, logistic regression, RNN/LSTM.

Course link for error bars: §9 covers the binomial estimator and the Clopper-Pearson interval — the
frequentist confidence interval on a test error rate. It assumes independent test samples, so with our
correlated windows it would be far too narrow (see Phase 2).

### Why not (high-degree) polynomial regression
Polynomial regression and ridge are not alternatives: the **feature map** φ(x) decides what the model can
represent (φ(x) = x → hyperplane; φ(x) = all monomials up to degree p → polynomial), **ridge** is how the
weights are fitted (least squares + λ‖w‖²) for any φ. Polynomial regression with regularization = ridge on
polynomial features; kernel ridge with the polynomial kernel (1 + xᵀx′)ᵖ = the same model, computed differently.

The problem is computing φ explicitly: with d = 54 inputs there are C(54 + p, p) monomials.

| degree p | features | explicit train matrix (70k windows) |
|---|---|---|
| 1 | 55 | 30 MB |
| 2 | 1,540 | 0.9 GB — fine |
| 3 | 29,260 | 16 GB + a 29k × 29k system — too much |
| 10 | ≈ 1.5 × 10¹¹ | impossible |

Curse of dimensionality (course §2): in 1-D, degree 10 is 11 coefficients; in 54-D it explodes.
**Kernel trick** (§4): the cost depends on the number of samples n (an n × n system), not on the number of
features — same cost for any degree, even infinite (RBF). Hence kernel ridge instead of explicit
high-degree polynomials (trained on a subsample because of the n³ cost).

Nuances: low degrees are fine explicitly (degree 2 = 1,540 features is tractable and contains `v_z²`, the
fix suggested by the scatter plot) — a possible step between ridge and kernel ridge. And higher degree is
not automatically better: huge variance and wild extrapolation; RBF kernels are smoother and more local.

### Core path
1. ✅ Majority baseline.
2. ✅ Ridge on A → B → C (input ladder): 0.840 → 0.930 → 0.946.
3. ✅ Kernel ridge on the best input (non-linear boundary): 0.956.
4. ✅ Linear SVM and RBF SVM on the same input (loss) → 2×2 with ridge / kernel ridge: boundary matters, loss only with full data.
5. ✅ Trees on the same input, 70k windows: decision tree → random forest → gradient boosting, with feature importance.
   Tree 0.940 → random forest 0.967 → gradient boosting **0.972 (best)**; `gb_10k` 0.962 > kernel ridge 0.956 at equal data.

### Weak spots to check (questions the professor could ask)
- **Random forest: is 200 trees enough?** `n_estimators` was fixed, not tuned (more trees only lower the variance, never
  hurt). Unchecked. Check (val only): fit ~800 trees per leg on train, score val with the first 25/50/…/800 trees
  (forest prediction = average over trees → one fit gives the whole curve); expect a plateau by ~100–200.
- **Real-time cost of gradient boosting**: 800 trees × 4 legs per timestep at 1 kHz (1 ms budget). Not measured.
  Check: time single-window and batch predictions of the saved models (`gb`, `rf`, `krr`, `svm_rbf`) on the test features.

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
