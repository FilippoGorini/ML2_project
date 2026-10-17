# Evaluation and windowing conventions in learning-based legged-robot contact estimation (Mini-Cheetah contact dataset)

Sources read in full text (arXiv PDFs via pdftotext) and code read directly from cloned GitHub repos (HEAD as of 2026-10-10):
- Lin, Zhang, Teng, Ghaffari et al. 2021, CoRL 2021 — https://arxiv.org/abs/2106.15713 ; code https://github.com/UMich-CURLY/deep-contact-estimator
- Ordonez-Apraez, Martin, Agudo, Moreno-Noguer 2023, RSS 2023 — https://arxiv.org/abs/2302.10433 ; code https://github.com/Danfoa/MorphoSymm
- Butterfield et al. 2024, MI-HGNN (ICRA 2025) — https://arxiv.org/abs/2409.11146 ; code https://github.com/lunarlab-gatech/Morphology-Informed-HGNN
- Xie, Gan et al. 2024, MS-HGNN — https://arxiv.org/abs/2412.01297
- Fischer Abati et al. 2025/2026, "Proprioceptive Image" — https://arxiv.org/abs/2510.14612
- Camurri et al. 2017, RA-L — https://www.robots.ox.ac.uk/~mobile/Papers/2017RAL_camurri.pdf
- Rotella, Schaal, Righetti 2018, ICRA — https://arxiv.org/abs/1709.07472

## Q1. Lin et al. 2021 (CoRL): split, window length, labels, test sequences, accuracy

### Takeaway
Lin et al. used a **hybrid**: a few whole sequences held out for testing, plus a **per-recording contiguous 70/15/15 time split** of the remaining recordings (no gap), with **stride-1 overlapping windows of 150 samples** labelled at the last timestep. It was not a random-window split; "shuffled" refers only to minibatch order. Window length 150 is asserted, not justified or ablated. Reported accuracy: 97.82% per leg and 93.88% on the 16-state metric.

### Cited Findings
- **Split (paper text):** "Some sequences are reserved for testing. The rest of the data are split into validation, testing, and training sets with the ratio of 15%, 15%, and 70%, respectively." — [Lin et al. 2021, §5](https://arxiv.org/abs/2106.15713)
- **Held-out test sequences named in the paper:** Table 2 compares methods on the "Concrete Short Loop", "Grass Test Sequence" and "Forest Test Sequence". Appendix Table 6 then reports "test sets" accuracy per terrain for all terrains (air trotting, air pronking, asphalt road, concrete, forest, grass, middle pebble, small pebble, rock road, sidewalk). This means the test numbers mix the within-recording 15% test slices with held-out sequences. The paper does not state which recordings are fully held out. — [Lin et al. 2021, Table 2 and Appendix Table 6](https://arxiv.org/abs/2106.15713)
- **Split (code):** `mat2numpy_split()` takes a **contiguous** slice of each recording and does not shuffle samples. The **first** 15% goes to val, the **next** 15% to test and the **last** 70% to train: `cur_val = cur_data[:num_val]`, `cur_test = cur_data[num_val:num_val+num_test]`, `cur_train = cur_data[num_val+num_test:]`. There is no gap between blocks. The slices from each recording are then `np.vstack`-ed into a single train.npy, val.npy and test.npy. Config: `train_ratio: 0.7`, `val_ratio: 0.15`, `window_size: 150`. — [deep-contact-estimator, utils/mat2numpy.py and config/mat2numpy_config.yaml](https://github.com/UMich-CURLY/deep-contact-estimator)
  - Conflicting description: Ordonez-Apraez et al. (2023) describe Lin's split as "the first 70% time-samples of each recording were assigned for training, the following 15% to validation, and the rest for testing". That is the opposite order to the public code (val, test, train). Both describe a contiguous per-recording split, not a random one. — [Ordonez-Apraez et al. 2023, Appendix "Details on dataset partitioning"](https://arxiv.org/abs/2302.10433) vs [deep-contact-estimator code](https://github.com/UMich-CURLY/deep-contact-estimator)
- **Windowing / stride:** "for data point at time n, we create a window with size w and append previous measurements within this window into a 2D array Dn … Dn is a w × 54 array. (In our case, we use w = 150.) Each time, the network takes Dn as input and estimates the contact state Sn as output." — [Lin et al. 2021, §4.1](https://arxiv.org/abs/2106.15713)
  - In the code the Dataset has `num_data = data.shape[0]-window_size+1` and `__getitem__(idx)` returns `data[idx:idx+window_size]` with label `label[idx+window_size-1]`. That gives **stride 1, maximally overlapping windows, labelled at the last sample**. Each window is z-scored with its own mean and std ("time-wise normalization"). The windows are built over the concatenated arrays, so some windows span the boundary between two recordings. — [deep-contact-estimator, utils/data_handler.py](https://github.com/UMich-CURLY/deep-contact-estimator)
- **Window-length justification:** the only justification given is "A window size of 150 is used to allow the network to infer from the time domain." There is no ablation of window length. The architecture ablation (Table 4) varies only the number of conv blocks. — [Lin et al. 2021, §5 and Table 4](https://arxiv.org/abs/2106.15713)
  - For the Cassie experiment: "Because data from Cassie is recorded at a higher frequency, a window size of 600 is used." "Two sequences are reserved as test sets, and two are used in training." — [Lin et al. 2021, §5.4](https://arxiv.org/abs/2106.15713)
- **Shuffling:** "We shuffled the training data to ensure generalizability in the model and reduce overfitting. We set the batch size to 30 and use 10−4 as the learning rate … 30 epochs." The code sets `shuffle: True` on the DataLoader, which shuffles minibatch order **after** the split. — [Lin et al. 2021, §5](https://arxiv.org/abs/2106.15713); [config/network_params.yaml](https://github.com/UMich-CURLY/deep-contact-estimator)
- **Data and labels:** "The IMU measurements are received at 1000 Hz, while other data are recorded at 500 Hz. All data are upsampled to match the IMU frequency." "Around 1,000,000 data points are collected from various terrains". Labels come from an offline algorithm that "applies a low-pass filter to the signal and extracts local minima and maxima using the future and past data points around the current timestamp. Hence, it cannot run online on the robot." — [Lin et al. 2021, §4.3](https://arxiv.org/abs/2106.15713)
- **Accuracy:** "On average, the trained network achieves 93.88% accuracy in 16 contact states and 97.82% in each leg individually … FPR of 1.82% and an FNR of 2.88% on the test sets." Held-out sequences: concrete short loop 98.18%, grass 97.78% and forest 97.08% leg-average accuracy. — [Lin et al. 2021, §5.1 and Table 2](https://arxiv.org/abs/2106.15713)
- **Generalization claim:** the abstract says "The trained network can estimate contact events on different terrains". There is no experiment that holds out a whole terrain or gait. — [Lin et al. 2021](https://arxiv.org/abs/2106.15713)

### Inferences
- The student's protocol (a) is the same **kind** of split as Lin et al.'s within-recording protocol: contiguous per-recording blocks with ratios 70/15/15. It differs in block order (Lin's code: val | test | train) and in adding **gaps** (Lin has none). The student's version is the more conservative of the two.
- Lin trained with stride 1. The student's stride 10 cuts redundancy by 10x but is not a published convention. The published convention is stride 1 with fully overlapping windows.
- With stride 1 and no gap, Lin's val and test windows that start near a block boundary share up to 149 samples of input with neighbouring blocks. The paper does not discuss this.

### Gaps
- The paper does not say which whole sequences were "reserved for testing", beyond the three named in Table 2.
- I could not confirm whether the public `mat2numpy_split` code is the exact version used for the paper's numbers. The block order differs from the ECNN paper's description.

## Q2. Ordonez-Apraez et al. 2023 (RSS; MorphoSymm/ECNN): the "modified" split

### Takeaway
ECNN replaced Lin's per-recording 70/15/15 split with a **recording-level hold-out**. Five recordings were chosen at random for the test set (air jumping gait, concrete pronking, concrete right circle, forest, small pebble). The other 10 recordings are split per recording into a contiguous first 85% for train and last 15% for val. The stated reason is to measure generalization and out-of-training-distribution performance. Window = 150, carried over from Lin, with stride-1 windows in the code.

### Cited Findings
- **Exact text:** "We modified the original dataset partitioning to properly evaluate the generalization capacity of the models. The original dataset was composed of 15 different recordings varying ground type and gait type … (most recordings were performed on a trot gait, which heavily biased the dataset to contact states 0, 6, and 9 …). The authors of [14] partitioned all 15 recordings into (70%, 15%, 15%) training, validation and testing … Because we are interested in studying the generalization capacity of the models and the out-of-training-distribution performance, we modified this partitioning such that among the 15 different recordings we selected randomly 5 recordings for testing, and the remaining 10 recordings were used for training splitting these recordings into (85%, 15%) training and validation splits as in [14], that is, for each recording, the first 85% data-samples go for training and the remaining for validation." — [Ordonez-Apraez et al. 2023, Appendix C-D2](https://arxiv.org/abs/2302.10433)
- **Recordings:** train/val = "air walking gait, concrete difficult slippery, concrete left circle, middle pebble, rock road, asphalt road, concrete galloping, grass, old asphalt road, sidewalk". Test = "air jumping gait, concrete pronking, concrete right circle, forest, small pebble". This is identical to the student's protocol (b). — [Ordonez-Apraez et al. 2023, Appendix](https://arxiv.org/abs/2302.10433)
- **Window:** inputs "captured over a history of 150 time-frames". Each sample is "a history of 150 samples y = {z_i} ∈ R^{54×150}". No justification is given beyond following Lin et al. — [Ordonez-Apraez et al. 2023, §VI-B and Appendix](https://arxiv.org/abs/2302.10433)
- **Code:** `cfg/dataset/contact.yaml` has `window_size: 150`, `train_ratio: 0.85`, `val_ratio: 0.15`, `test_ratio: 1`, `shuffle: True`, `batch_size: 30`. The comment reads: "Dataset is already partitioned in training-val data and test data separatedly". `load_and_split_mat_files()` takes contiguous slices of each recording (`cur_data[lower_lim : lower_lim + partition_size]`) and vstacks them. The Dataset uses `num_data = data.shape[0] - window_size + 1`, i.e. **stride 1**, with **no gap** between train and val. — [MorphoSymm, morpho_symm/data/contact_dataset/umich_contact_dataset.py and cfg/dataset/contact.yaml](https://github.com/Danfoa/MorphoSymm)
- **Purpose:** the paper notes that the test set exposes the dataset's bias toward trot contact states. Results are reported as mean ± std over 8 seeds, with sample-efficiency curves. — [Ordonez-Apraez et al. 2023, Fig. 4 caption](https://arxiv.org/abs/2302.10433)

### Inferences
- "Selected randomly" refers to choosing **recordings** at random, not windows. It is a single fixed draw, with no cross-validation over recordings.
- Under this protocol, validation (the last 15% of each training recording) is in-distribution, while test is out-of-distribution. This is the same structure as the student's protocol (b).

### Gaps
- The paper does not say why 85/15 was used instead of keeping 70/15, and does not give a seed for the random choice of recordings.

## Q3. MI-HGNN (Butterfield et al., ICRA 2025) and MS-HGNN (Xie et al. 2024)

### Takeaway
Both papers adopt the ECNN recording-level split and the 150-sample history explicitly "for a fair comparison". MI-HGNN's code splits train and val at the **window index**, with stride 1 and no gap. Neither paper ablates window length or discusses overlap.

### Cited Findings
- **MI-HGNN split:** "We use the modified train/val/test data split introduced in [27] throughout this experiment, in which the test set includes recordings on unseen ground types and gait types, to evaluate the generalization ability of our model on unseen scenarios." — [Butterfield et al. 2024, §V-A](https://arxiv.org/abs/2409.11146)
- **MI-HGNN window:** "Implementation: For a fair comparison, we follow the experiment setup in [17], [27]. The input data for all baseline models is a history of 150 samples x = {z_i}_{i=0}^{150} ∈ R^{54×150}". All models are trained with batch size 30, "use time-wise normalization per dataset entry, and employ an early-stopping mechanism monitoring validation loss." — [Butterfield et al. 2024](https://arxiv.org/abs/2409.11146)
- **MI-HGNN code:** `research/train_classification.py` sets `history_length = 150`. For each of the 10 train/val recordings it computes `split_index = int(np.round(dataset.__len__() * 0.85))` and assigns windows `[0, split_index)` to train and `[split_index, len)` to val. The test set is a ConcatDataset of concrete_pronking, concrete_right_circle, small_pebble, air_jumping_gait and forest. A helper `ensure_dataset_splits_match_morphoSymm()` checks that the split matches MorphoSymm. The dataset length is `num_samples - history_length + 1`, which means **stride 1**. Each window is labelled with the contact at its last row: `contacts[seq_num:seq_num+history_length][-1]`. — [Morphology-Informed-HGNN, research/train_classification.py, src/mi_hgnn/datasets_py/{flexibleDataset,LinTzuYaunDataset}.py](https://github.com/lunarlab-gatech/Morphology-Informed-HGNN)
- **MI-HGNN results (unseen test set, 8 runs):** "our MI-HGNN achieves a mean accuracy of 87.7% across 8 runs, improving the performance of the second best model (i.e., ECNN with 78.8% mean accuracy) … Compared with the morphology-agnostic CNN model (with 71.8% mean accuracy) … As the evaluation is conducted on out-of-training-distribution data, our MI-HGNN model shows superior generalization capability". These are 16-state accuracies. — [Butterfield et al. 2024, §V-A Results](https://arxiv.org/abs/2409.11146)
- **MS-HGNN:** "Following the data split from Apraez et al. (2023), we use the same unseen sequences for testing and allocate 85% and 15% of the remaining data for training and validation. As test sequences contain unseen gait/terrain combinations, this setup is also helpful for evaluating model's generalization ability on out-of-distribution data. … The input comprises 150-step histories of [q, q̇, a_b, ω_b, p, v] ∈ R^54". MS-HGNN-K4 is reported to improve 16-state accuracy by 11% over ECNN, and MS-HGNN-C2 to beat MI-HGNN (accuracy 0.875 vs 0.870). — [Xie et al. 2024](https://arxiv.org/abs/2412.01297)
- **Proprioceptive Image (2025/26), a later work on the same dataset:** uses a **w = 10** sliding window ("15 times shorter window size") and reports 94.5% 16-state accuracy on the unseen test set, compared with 87.7% for MI-HGNN. "Unlike the original setup, however, our model was trained on the merged training and validation splits (treated as a single training set) and evaluated on the merged test splits." The paper also reports per-terrain comparisons against Lin et al. (Table I). — [Fischer Abati et al., arXiv 2510.14612](https://arxiv.org/abs/2510.14612)

### Inferences
- In MI-HGNN, splitting at the window index with stride 1 and no gap means the first val window shares 149 of its 150 input samples with the last train window. Only one boundary per recording is affected, so the effect is negligible, but it is unprotected. The student's gap is stricter.
- The numbers show how much the protocol matters. On the unseen-recording protocol, Lin's CNN reaches about 71.8% 16-state accuracy (MI-HGNN's re-run). On Lin's own protocol, which includes within-recording test slices, the same model reaches 93.88%. Training data and implementations differ between the two, so this is not a controlled comparison, but it suggests that within-recording test scores are optimistic about generalization to new recordings.
- PI's w = 10 result suggests that 150 samples is not necessary for this task. It is a convention inherited from Lin.

### Gaps
- Neither HGNN paper discusses stride or overlap. The stride-1 detail comes only from the MI-HGNN code; I did not inspect the MS-HGNN code.

## Q4. Other contact-estimation works: evaluation splits

### Takeaway
Earlier model-based and probabilistic contact estimators are fit on small amounts of data. Where they describe a split, it is a **held-out portion of a log or a separate log**, not random samples. None discuss leakage.

### Cited Findings
- **Camurri et al. 2017 (RA-L, HyQ):** contact probability comes from a logistic regression on the normal GRF, fit "on a training set of data collected from characteristic motions". For real data: "we performed training on half of a trot log and half of a crawl log from our dataset … and we used the rest of the dataset as a test set". That is a temporal/log-level hold-out with no windows. — [Camurri et al. 2017](https://www.robots.ox.ac.uk/~mobile/Papers/2017RAL_camurri.pdf)
- **Rotella et al. 2018 (ICRA):** unsupervised fuzzy c-means on wrench and IMU features. "Each data point x_k ∈ R^{7T} is a time series of the past T = 20 samples … We include a short time-history to improve estimation response time; optimization of this time window is left to future work." Validation is done in a simulated state estimator. — [Rotella, Schaal, Righetti 2018](https://arxiv.org/abs/1709.07472)
- **Lin et al. 2021, Cassie experiment:** "Two sequences are reserved as test sets, and two are used in training" (sequence-level). — [Lin et al. 2021](https://arxiv.org/abs/2106.15713)

### Inferences
- Across this literature, held-out logs or sequences are the norm whenever a test set is described. I found no paper that split contact data by randomly sampling windows.

### Gaps
- I did not read Bloesch/Hwangbo et al. 2018 ("Probabilistic foot contact estimation", HMM-based, ANYmal), Hartley et al. 2020 IJRR (contact-aided InEKF, which uses contact sensors rather than a learned estimator, so no ML split is expected), Piperakis et al., or Kim et al. Their split conventions are **not verified** here.

## Q5. Window-length ablations or justifications for 150 on this dataset

### Takeaway
None of the papers I read ablate the window length on the Mini-Cheetah dataset. 150 comes from Lin et al. with a one-line rationale and is carried forward "for a fair comparison".

### Cited Findings
- Lin: "A window size of 150 is used to allow the network to infer from the time domain." For Cassie, "a window size of 600 is used" because its data is recorded at a higher frequency. — [Lin et al. 2021](https://arxiv.org/abs/2106.15713)
- ECNN, MI-HGNN and MS-HGNN reuse 150 samples. MI-HGNN says explicitly "For a fair comparison, we follow the experiment setup in [17], [27]". — [Butterfield et al. 2024](https://arxiv.org/abs/2409.11146); [Ordonez-Apraez et al. 2023](https://arxiv.org/abs/2302.10433); [Xie et al. 2024](https://arxiv.org/abs/2412.01297)
- PI uses w = 10 and reports higher accuracy than the w = 150 baselines. — [Fischer Abati et al. 2510.14612](https://arxiv.org/abs/2510.14612)

### Inferences
- At 1 kHz, 150 samples is 150 ms. At typical Mini-Cheetah trot frequencies this is a fraction of a gait cycle up to roughly one cycle, but **no paper states this rationale**. Treat it as the student's own argument if used.

### Gaps
- I did not check whether the PI paper contains a window-size ablation of its own.

## Q6. Explicit discussion of temporal leakage, overlapping windows, or random vs sequence-level splits

### Takeaway
No paper I read uses the word "leakage" or discusses window overlap across split boundaries. The only explicit discussion of split design is ECNN's motivation for holding out whole recordings: measuring generalization and out-of-distribution performance.

### Cited Findings
- ECNN's rationale is "properly evaluate the generalization capacity of the models" and "out-of-training-distribution performance". It does not mention leakage. — [Ordonez-Apraez et al. 2023](https://arxiv.org/abs/2302.10433)
- MI-HGNN: "the test set includes recordings on unseen ground types and gait types, to evaluate the generalization ability of our model on unseen scenarios." — [Butterfield et al. 2024](https://arxiv.org/abs/2409.11146)
- All three codebases (Lin, MorphoSymm, MI-HGNN) use stride-1 windows and contiguous splits **without gaps**. Lin and MorphoSymm build windows over vstacked arrays, so some windows span the boundary between recordings. — code links in Q1–Q3

### Inferences
- **Nothing contradicts the student's choices.** No published work used a random-window split. Both the student's protocol (a) and protocol (b) match published conventions, (a) following Lin and (b) following ECNN, MI-HGNN and MS-HGNN. The student's gaps add protection the original code lacks. The stride differs: published work uses stride 1, the student uses stride 10 for training. That is a defensible efficiency choice, but it should be stated as a deviation.
- Protocol (b) numbers can be compared directly with ECNN, MI-HGNN, MS-HGNN and PI (16-state accuracy on the 5 unseen recordings). Protocol (a) numbers are roughly comparable with Lin's 93.88% / 97.82%, with caveats: Lin's block order is different, there are no gaps, and Lin's test set also includes whole held-out sequences.

### Gaps
- I found no paper in this subfield that compares random-window vs sequence-level splits empirically.
