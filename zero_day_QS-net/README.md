# Zero-Day Attack Detection with a Noisy Variational Quantum Classifier + Conformal Prediction

A hybrid classical–quantum pipeline that detects known network-attack classes **and** flags previously-unseen ("zero-day") attacks in an IoT/logistics network traffic dataset. The core idea is to represent each traffic sample as a **quantum density matrix**, learn a per-class "prototype" state for every known attack category, and use **split conformal prediction** on quantum state fidelity to decide, with a statistical guarantee, whether a new sample belongs to a known class or should be rejected as an anomaly.

---

## 1. Abstract

Classical intrusion-detection classifiers are good at recognizing attack types they were trained on, but they degrade unpredictably on **zero-day attacks** — traffic patterns that didn't exist in the training data. Most classifiers will happily (and wrongly) assign a zero-day sample to whichever known class it resembles most, with no reliable signal that the prediction is untrustworthy.

This project explores whether a **quantum feature map**, combined with **conformal prediction** (a distribution-free statistical framework for building prediction sets with guaranteed error rates), can give a principled "I don't recognize this" signal — instead of just picking a best guess and being silently wrong.

Concretely, the notebook builds a pipeline that:
1. Deliberately **holds out the rarest attack class** in the dataset and pretends it doesn't exist during training (this stands in for a genuine zero-day).
2. Trains a noisy variational quantum circuit to map every known-class sample to a compact quantum state, encouraging same-class states to cluster and different-class states to separate.
3. Calibrates a **rejection threshold** on a held-out calibration set so that, at test time, the model flags out-of-distribution inputs as `ZERO_DAY` at a controlled false-alarm rate.
4. Evaluates whether the held-out "zero-day" class actually gets caught by that rejection mechanism.

---

## 2. Dataset

**Source:** [`zero-day-attack-detection-in-logistics-networks`](https://www.kaggle.com/datasets) on Kaggle (`zero_day_attack_detection_dataset_V1-400k.csv`), pulled via `kagglehub`. The notebook works on a 500-row sample of the ~400k-row file for fast iteration; the pipeline is designed to scale to the full dataset.

Each row represents a network/logistics session and includes fields such as:
- Network/session metadata: `Time`, `Session ID`, `IP Address`, `Geolocation`, `User-Agent`
- Traffic statistics: `Response Time` (ms), `Data Transfer Rate` (Mbps), `Protocol`, `Flag`
- Application/context fields: `Application Layer Data`, `Event Description`, `Family`
- Labels: `Threat Level` (multi-class attack/benign label used as the classification target) and `Prediction`

### Preprocessing
- High-cardinality / free-text / identifier columns that don't generalize (`Time`, addresses, session/user-agent/geolocation, raw application-layer text, event descriptions) are dropped.
- `Response Time` and `Data Transfer Rate` are stripped of their units (`" ms"`, `" Mbps"`) and cast to float.
- Remaining categorical columns (`Protocol`, `Flag`, `Family`, `IP Address`, `Threat Level`, `Prediction`) are label-encoded.

### Simulating a zero-day
The **rarest class** in `Threat Level` (by row count) is held out entirely as the "zero-day" class:
- `known_df` — every class except the rarest → used for train/calibration/test
- `zeroday_df` — only the rarest class → never seen during training, used only to check whether the model correctly rejects it later

The known data is then split into:
| Split | Purpose |
|---|---|
| **Train** (60%) | fit the quantum circuit parameters |
| **Calibration** (20%) | compute the conformal rejection threshold |
| **Test** (20%) | evaluate accuracy and false-alarm rate on known classes |

All splits are stratified by class.

---

## 3. Classical feature selection → quantum encoding

Qubits are an expensive, noisy resource in near-term quantum computing, so the feature space is compressed down to exactly **6 features → 6 qubits** before anything touches the quantum circuit:

1. A `RandomForestClassifier` (300 trees) is trained on the known-class training data.
2. The **top-6 features by Gini importance** are kept; everything else is dropped.
3. Features are standardized (`StandardScaler`, fit on train only).
4. Standardized values are **clipped to [-3, 3]** (to bound outliers) and linearly rescaled to **[0, π]** — this is the "angle encoding" step, since rotation gates like `RY(θ)` expect an angle:

   ```
   angle = (clip(x, -3, 3) + 3) / 6 * π
   ```

This gives one bounded rotation angle per feature per qubit, which is exactly what the quantum circuit's data-encoding gates consume.

---

## 4. The quantum circuit

Built in **PennyLane** on the `default.mixed` simulator (a density-matrix simulator, chosen deliberately — see below), with `N_QUBITS = 6` and `N_LAYERS = 4`.

### 4.1 Data re-uploading encoder
Rather than encoding the input once at the start, the 6 feature angles are **re-applied at every layer** (`data_reupload_encoder`). This "data re-uploading" trick is a well-known way to increase the expressivity of a shallow variational circuit without adding qubits — each layer gets another look at the raw data interleaved with trainable rotations.

### 4.2 Variational block (the trainable part)
Each layer applies, per qubit:
- `RY(θ_layer,qubit,0)`
- `RZ(θ_layer,qubit,1)`

followed by a **ring of CNOTs** (`qubit → qubit+1 mod N_QUBITS`) to entangle all qubits, including wrapping the last qubit back to the first.

### 4.3 Noise channel
After every layer, a `DepolarizingChannel(p=0.01)` is applied to every qubit. This isn't a simulation artifact to be tolerated — it's part of the design: the depolarizing noise makes the resulting state a genuine **mixed state** (density matrix ρ, not a pure state vector), which is required for the noise-aware robustness bound used later in inference (§6). Modeling noise explicitly is also more representative of what would happen on real quantum hardware.

### 4.4 Two circuit "views" of the same architecture
- **`rho_circuit(x, θ)`** → returns the full 6-qubit **density matrix** `qml.density_matrix(...)`. Used for prototype construction, fidelity/trace-distance computations, and conformal scoring.
- **`readout_circuit(x, θ)`** → returns per-qubit `⟨PauliZ⟩` expectation values. Used purely to feed a small linear classifier head for the cross-entropy training signal.

---

## 5. Training objective: a 3-term hybrid loss

Parameters trained: the circuit's `θ` tensor (shape `[N_LAYERS, N_QUBITS, 2]`) and a linear `classifier_head` (`nn.Linear(N_QUBITS, n_classes)`) on top of the Pauli-Z readout, both optimized jointly with Adam (`lr=0.05`).

For every mini-batch:

**1. Cross-entropy loss (`L_CE`)** — standard supervised classification loss on the linear head's logits, computed from the Pauli-Z expectation readout.

**2. Intra-class fidelity loss (`L_intra`)** — for each class present in the batch, compute a **prototype density matrix** (the mean of that class's density matrices in the batch), then penalize `1 - fidelity(ρ_sample, ρ_prototype)` for every sample against its own class's prototype. This pulls same-class quantum states toward a shared "cluster center" in state space.

**3. Inter-class trace-distance loss (`L_inter`)** — for every pair of classes present in the batch, subtract their trace distance from the loss (i.e., reward greater distinguishability between class prototypes). This pushes different classes' prototype states apart.

**Combined loss:**

```
L = L_CE + λ1 · L_intra + λ2 · L_inter        (λ1 = 0.5, λ2 = 0.1)
```

The gradient variance of `θ` is logged every epoch (`grad_var`) — a standard diagnostic for detecting **barren plateaus**, a well-known trainability failure mode in variational quantum circuits where gradients vanish exponentially with qubit count/depth.

After training, **final class prototypes** are recomputed once over the *entire* training set (not just the last batch) by averaging each class's density matrices — these are the reference states used at inference time.

---

## 6. Conformal prediction: turning fidelity into a rejection rule

This is the statistical core of the "zero-day detection" claim, and it's built in two stages.

### 6.1 Nonconformity score
For any sample, its **nonconformity score** is:

```
s(x) = 1 − max_c fidelity(ρ(x), prototype_c)
```

i.e., one minus its quantum fidelity to the *closest* known-class prototype. Low `s` = looks like a known class; high `s` = doesn't resemble anything the model has seen.

### 6.2 Split-conformal calibration
The nonconformity score is computed for every sample in the **held-out calibration set** (samples the circuit never trained on, but which are still known-class). Given a target miscoverage rate `ALPHA = 0.05`, the conformal quantile is:

```
q = the ⌈(1 − α)(n_cal + 1)⌉-th smallest calibration score
```

This is the standard split-conformal-prediction quantile formula, and it comes with a distribution-free guarantee: under exchangeability, a *true* known-class point will exceed this threshold (and thus get rejected) at most `α` of the time, regardless of what the underlying quantum model's decision boundary actually looks like. `q` is the single number that separates "confidently known" from "reject as zero-day."

### 6.3 Lipschitz robustness estimate
Before running inference, this estimates the **Lipschitz constant** of the quantum feature map — how much the output density matrix can change for a small perturbation `δ` in the input angles:

```
L_φ ≈ max over probes of  trace_distance(ρ(x), ρ(x + δ·noise)) / δ
```

This is estimated empirically by perturbing 30 random training points and taking the worst-case ratio observed. `L_φ` is later used to convert a fidelity margin into an interpretable **confidence radius**.

### 6.4 Unified inference rule
For a new sample `x`:
1. Compute fidelity to every class prototype.
2. If the nonconformity score `s = 1 − max fidelity` exceeds the calibrated threshold `q` → **reject as `ZERO_DAY`**.
3. Otherwise, predict the closest-fidelity class `c*`, and report a confidence radius:

   ```
   R = (fidelity_1st − fidelity_2nd) / (2 · (1 − p) · L_φ · C_f)
   ```

   where `p` is the depolarizing noise probability and `C_f` is a fixed constant (currently `1.0`). Intuitively, `R` combines the margin between the best and second-best class match with how noise-sensitive (`p`) and input-sensitive (`L_φ`) the circuit is — a small margin or a jumpy feature map both shrink the reported confidence.

---

## 7. Evaluation

On the held-out **known** test set:
- **False-zero-day rate**: fraction of genuinely-known test samples incorrectly rejected as `ZERO_DAY`. This should track close to the target `ALPHA` (5%) — the conformal guarantee is checked empirically here.
- **Balanced accuracy**: computed only over the non-rejected predictions, averaged per class (robust to class imbalance, unlike raw accuracy).
- **Confusion matrix**: includes a `-1` row/column representing rejected (`ZERO_DAY`-flagged) predictions, so rejections are visible alongside class-vs-class confusions.

On the held-out **zero-day** set (the truly unseen class):
- **Zero-day detection rate**: fraction of the held-out rare-class samples that get correctly flagged as `ZERO_DAY` by the same conformal threshold. This is the key number for the project's central question — does the conformal rejection rule generalize from "rejecting known-class outliers" to "rejecting an entirely new class it never saw"?

---

## 8. Project structure

```
zero-day-qml.ipynb
├── 1. Data loading           — kagglehub download, CSV read (500-row sample)
├── 2. Preprocessing          — drop/clean columns, label-encode categoricals
├── 3. Zero-day split         — hold out rarest class, train/cal/test split
├── 4. Feature selection      — Random Forest top-6 features → angle encoding
├── 5. Quantum circuit        — data re-uploading VQC, ring CNOT, depolarizing noise
├── 6. Training loop          — hybrid CE + intra-fidelity + inter-trace-distance loss
├── 7. Conformal calibration  — nonconformity scores, split-conformal threshold q
├── 8. Lipschitz estimation   — robustness constant L_φ via input perturbation
├── 9. Unified inference      — fidelity-based classification + ZERO_DAY rejection
└── 10. Evaluation            — false-zero-day rate, balanced accuracy, confusion
                                 matrix, zero-day detection rate
```

---

## 9. Requirements

```bash
pip install pennylane --upgrade
pip install pandas numpy scikit-learn torch kagglehub
# optional, for GPU-accelerated simulation:
pip install pennylane-lightning[gpu]
```

A CUDA-capable GPU is used automatically if available (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`); the notebook falls back to CPU otherwise. Note that `default.mixed` (density-matrix simulation) is significantly more memory- and compute-intensive than a pure-state simulator, since it scales as `O(4^n)` rather than `O(2^n)` in the number of qubits — this is why `N_QUBITS` is kept small (6).

---



## 10. limitations - for the traing phase

- The notebook currently runs on a **500-row sample**; results should be re-validated on a larger slice (or the full ~400k rows) before drawing strong conclusions, especially since the "zero-day" class is by definition the rarest and calibration/test splits of it can be very small.
- Only one class is ever held out as "zero-day" — the true rarest class in this fixed sample. 
- `L_intra`/`L_inter` are computed **per-batch**, not over the full dataset, so class prototypes used during training can be noisy estimates for small batch sizes (`BATCH = 10`).
- `C_f` in the confidence-radius formula is currently a fixed placeholder (`1.0`) 
- Only 5 training epochs are configured — this is convenient for quick iteration but likely under-trained for a production-quality result; gradient variance logging is in place to help detect barren-plateau issues if depth/qubit count are scaled up.
- by running 10000 data rows , it take more time so that i uses limited data and epoch for this architechture , many fixes required in random number or values generation for theta values etc.

---


