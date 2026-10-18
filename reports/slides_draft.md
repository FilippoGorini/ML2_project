# Slide text draft

Text for the slides in `PROGRESS.md` "Presentation plan", ready to paste into the template. Per slide: **title**,
on-slide text (short), figure, *notes* (what to say). Numbers: `notebooks/05_presentation.ipynb`.

---

## 1. Title
**Foot-contact estimation for the Mini-Cheetah**
Classical models on proprioceptive time windows
Ridge · Kernel ridge · Decision tree · Random forest · Gradient boosting

---

## 2. The problem
**Is each foot on the ground right now?**
- Input: the last 150 ms of proprioception (joints, IMU, feet). Output: contact yes / no, for each of the 4 feet.
- Use: the contact-aided EKF (state estimation) treats a foot in contact as fixed on the ground.
- Two errors, two costs:
  - **false contact** (the foot is swinging): wrong velocity enters the filter → the estimate drifts. **Harmful.**
  - **missed contact**: the filter skips one measurement. **Mild.**

*Notes: this asymmetry comes back in the metric (slide 7) and every model slide reports the false-contact rate.*

---

## 3. The dataset
**MIT Mini-Cheetah contact dataset** (Lin et al., CoRL 2021)

| | |
|---|---|
| recordings | 15: 8 terrains, 3 gaits (trot, pronk, gallop) + 2 with the robot held in the air |
| sampling | 1 kHz |
| timesteps | 1,008,941 (~17 min) after dropping the last 300 ms of each recording |
| channels | 54: joint angles q (12), joint velocities q̇ (12), IMU acceleration (3) and angular velocity (3), foot positions p (12) and velocities v (12) |
| labels | contact of the 4 feet, computed offline from foot height (an estimate, not force sensors) |

Figures: terrains photo = **Fig. 3(b) of Lin et al.** (arXiv 2106.15713, cite it under the picture); `data_labels`.

*Notes: the labels use future samples (local minima of the filtered foot height), so they are themselves an estimate;
the last 300 ms are dropped because the labels stop there while the robot is still walking.*

---

## 4. Sample and task
**One sample = one 150 ms window**
- Window of the last 150 timesteps × 54 channels, as in the original papers.
- Label = contact state at the window's **last** timestep → exactly what the robot would ask at run time.
- **4 independent binary classifiers, one per leg** (contact = positive class).

Figure: sketch of a window sliding over the signal (to draw in the template).

---

## 5. Split
**Split each recording in time, not at random**

Every recording, separately: `[ train 70% | gap | val 15% | gap | test 15% ]` → every recording and terrain appears
in all three sets.

- Consecutive windows are near-copies (they share 149 of 150 timesteps): a random split would put near-copies of
  training windows in the test set → optimistic score.
- Contiguous blocks + **one-window gaps (150 ms)**: no timestep is shared between two sets.
- Overlapping windows inside each set; train / val keep every 10th window (same result as all, 10× cheaper):

| set | windows (stride 1) | windows used |
|---|---|---|
| train | 704,688 | 70,476 (stride 10) |
| val | 148,755 | 14,882 (stride 10) |
| test | 148,763 | 148,763 (every timestep) |

Figure: `split_one_recording`.

*Notes: references for the optimism of random splits on time series: scikit-learn CV guide, Arlot & Celisse 2010.*

---

## 6. Tuning and normalization
**Choose on validation, refit, test once**

```python
search = GridSearchCV(make_pipeline(StandardScaler(), model), grid,
                      cv=PredefinedSplit(train=-1, val=0),   # one fixed train -> val fold
                      scoring="f1_macro", refit=True)
search.fit(X_train_val, y_train_val)                          # refit on train + val
search.predict(X_test)                                        # test used once
```

Flow:
1. for every grid point: scaler **fitted on train** → model fitted on train → scored on **val** (scaler only applied);
2. best grid point → scaler and model **refitted on train + val** (85,358 windows);
3. **test**: scaler and model only applied, once.

- One fixed validation block, no k-fold: random folds would leak near-copies, as in slide 5.
- Trees, forest and boosting take the features unscaled (threshold splits do not depend on the scale).

*Notes: the scaler is a pipeline step, so it is always fitted on exactly the data the model is fitted on; mean and
std never see val or test while tuning, nor test at the end.*

---

## 7. Metric and imbalance
**Accuracy is not enough**

| | F1 | accuracy | false contact | missed contact |
|---|---|---|---|---|
| majority ("always air") | 0.000 | 0.665 | 0% | 100% |
| gradient boosting | 0.972 | 0.981 | 1.6% | 2.4% |

- Main metric: **F1 on contact, averaged over the 4 legs**; beside it the **false-contact rate** (false contacts /
  true air timesteps: the harmful error).
- Imbalance is mild (~34% contact); recall ≥ precision for every model → no bias towards "air".
- **No class weighting**: it would make the models say "contact" more often → more false contacts.

---

## 8. Ridge: which input?
**Ridge regression** — a linear score, contact if f(x) > 0

  f(x) = wᵀx + b,  min_w ‖Xw − y‖² + α‖w‖²   (y = ±1)

Same model, three inputs:

| input | values | test F1 |
|---|---|---|
| current timestep | 54 | 0.840 |
| raw window (15 log-spaced lags) | 810 | 0.930 |
| hand-crafted features | 390 | **0.946** |

- **History (+9 points)**: one instant is not enough (pronking 0.61 → 0.91).
- **Features (+1.7)**: per channel current value, changes over 5 / 50 ms, mean, std, min, max; per foot height above
  its window minimum and speed. Non-linear summaries a linear model cannot build itself.
- Lags and feature groups chosen on validation; features best → **the input for every model from here on**.

Figures: `input_comparison`, `why_history_pronking`.

*Notes: in pronking stance the foot is neither still nor at a fixed height, so a threshold on current values fails.*

---

## 9. Ridge on features
**Ridge: the linear limit**
- α searched over 10⁻³ … 10⁵: flat up to ~10³ → the model **under-fits**; the limit is the linear boundary, not
  the tuning. Chosen α = 0.1.
- Test F1 **0.946**, false contacts 3.2%, missed 4.6%.

Figures: `val_ridge_feat_all`, `cm_ridge_feat_all`.

---

## 10. Kernel ridge
**Same loss, curved boundary**

  k(x, x′) = exp(−γ‖x − x′‖²),  (K + αI) a = y,  f(x) = Σᵢ aᵢ k(xᵢ, x)

- RBF kernel = similarity between two windows; one weight aᵢ per **training window** instead of one per feature.
- n × n kernel matrix → trained on **10k windows** (all 70k would need 40 GB).
- α and γ interact (diagonal band of good values): searched together. Chosen α = 0.03, γ = 0.3 / 390.
- Test F1 **0.956** (+1.0 over ridge), false contacts 2.5%.

Figures: `val_krr_feat_all`, `cm_krr_feat_all`.

---

## 11. Decision tree
**A sequence of yes / no questions**
- Each split asks "feature ≤ threshold?", chosen to make the two groups purest (Gini); each leaf predicts its majority.
- Depth-3 example (RF leg): the first question is about the **LF** foot — trot diagonal pairs, found by the tree itself.
- Leaf size is the bias-variance knob: 1 window per leaf memorises, 200 too coarse; chosen 50.
- Test F1 **0.940**: below ridge → one tree has **high variance**.

Figures: `depth3_tree`, `val_tree_feat_all`.

---

## 12. Random forest
**Average many different trees → less variance**
- 200 deep trees per leg, each on a **bootstrap sample** of the windows, each split among a **random subset of
  features**; prediction = average of the trees.
- Smallest leaves best (leaf size 1, the opposite of one tree): each tree memorises, the average cancels it.
  `max_features` barely matters (chosen 39 of 390).
- Test F1 **0.967** (+2.7 over one tree), false contacts 1.8%.

Figures: `val_rf_feat_all`, `cm_rf_feat_all`.

---

## 13. Gradient boosting
**Add trees one at a time, each correcting the errors left → less bias**

  F(x) = Σₘ ν hₘ(x),  p(contact) = σ(F(x))

- Each new small tree hₘ is fitted to what the current ensemble still gets wrong (gradient of the log-loss),
  scaled by the learning rate ν = 0.1.
- Grid: number of trees × tree size; the whole grid is within 0.969–0.975 on validation. Chosen 800 trees × 63 leaves.
- Test F1 **0.972**, false contacts **1.6%** (ridge 3.2%).
- Fair check: on the same 10k windows as kernel ridge, 0.962 vs 0.956 → ~0.6 points from the model, ~1.0 from the data.

Figures: `val_gb_feat_all`, `cm_gb_feat_all`.

---

## 14. Model comparison
**All models, same input and split**

| | ridge | kernel ridge | tree | forest | boosting |
|---|---|---|---|---|---|
| test F1 | 0.946 | 0.956 | 0.940 | 0.967 | **0.972** |
| false contact | 3.2% | 2.5% | 3.2% | 1.8% | **1.6%** |

Figure: `model_comparison`.

---

## 15. Where the errors are
**Errors sit at touchdown and lift-off**
- 72% of boosting's errors are within 5 ms of a true touchdown / lift-off (ridge on one timestep: 27%).
- Sensor data update every ~2–5 ms, and the label is itself an estimate → mostly timing disagreements.

Figures: `error_timing`, `strips_trot`.

---

## 16. Unseen recordings (protocol B)
**The papers' split: 5 whole recordings never seen in training**
- Held out: air jumping, concrete pronking, concrete right circle, forest, small pebble. Pronking is the only
  pronking recording → **an unseen gait**.
- Boosting **0.938**; without pronking 0.970 (protocol A: 0.972) → a new terrain costs ~nothing, a new gait does.
- The extra errors are missed contacts (the mild kind): false contacts stay at 1.5%.

Figure: `protocol_b`.

---

## 17. Against published models
**Same split, same metric**

| model | F1 |
|---|---|
| CNN (Lin et al. 2021) | 0.861 |
| MI-HGNN (2024) | 0.931 |
| MS-HGNN (2024) | 0.939 ± 0.006 |
| **ours: gradient boosting** | **0.938** |

- No neural network, ~10× fewer training windows.
- Caveats: we drop the last 300 ms, refit on train + val, one run (theirs: mean of 4 seeds).

Figure: `papers_comparison`.

---

## 18. Limits
- Labels are an estimate (offline, from foot height): part of the remaining errors is label noise.
- Not run on the robot: ~15 ms per call in Python for the tree ensembles (budget 1 ms at 1 kHz); the computation
  itself fits, deployment would need compiled trees.
- One fixed validation block per recording.
- Unseen gaits remain hard (pronking 0.88–0.90).

---

## 19. Conclusion
- **Current timestep → history → features → tree ensembles: 0.84 → 0.97** on seen recordings; false contacts 1.6%.
- The best model's errors sit within a few ms of the label's transitions.
- **Unseen recordings: 0.938**, level with the best published model.
