# Methodology

A full technical account of the model: the physics embedding, the custom kernel, the fitting
procedure and the optimisation loop. For results see [`RESULTS.md`](RESULTS.md); for a
high-level summary see [`ABOUT.md`](ABOUT.md).

---

## 1. Data

### 1.1 Source and shape

`biogas_ml_dataset.csv` contains **40 usable observations** after `dropna()` (the raw file uses
blank rows as visual separators between blends). The design is a full crossing of
**4 feedstock blends × 10 digestion days**.

| Blend (Food / Weed) | TS (%) | VS (%) | pH | C:N | Yield range (cm³/day) |
|:---|:---:|:---:|:---:|:---:|:---:|
| 100 / 0 | 20.0 | 85.00 | 7.00 | 20.00 | 48 → 97 |
| 75 / 25 | 22.5 | 83.75 | 7.05 | 23.75 | 55 → 112 |
| 50 / 50 | 25.0 | 82.50 | 7.10 | 27.50 | 62 → 165 |
| 25 / 75 | 27.5 | 81.25 | 7.15 | 31.25 | 50 → 97 |

### 1.2 Structural note on the feature space

`TS`, `VS`, `C_N` and `pH` are **deterministic linear functions of the blend ratio**:

```
TS  = (20·Food% + 30·Weed%) / 100
VS  = (85·Food% + 80·Weed%) / 100
C_N = (20·Food% + 35·Weed%) / 100
```

The nominal seven-dimensional input space therefore has an **effective dimensionality of two**
(blend ratio and day). This matters for interpreting the results: the model is fitting a smooth
surface over two latent variables, not seven independent ones, which is why every kernel tested
performs near-identically.

### 1.3 Preprocessing

- `StandardScaler` fitted on all 40 rows, applied to all seven features.
- `train_test_split(test_size=0.2, random_state=42)` → **32 train / 8 test**.
- The fitted scaler is handed to the kernel via `set_scaler()`, so the kernel can invert the
  transform and evaluate the physics functions in physical units.

> ⚠️ **Protocol caveat.** The scaler is fitted before the split, and the split is random rather
> than grouped by blend. Since rows within a blend differ only in `Day`, each test point has
> near-duplicate neighbours in the training set. See [§6](#6-known-limitations).

---

## 2. The physics embedding φ(x)

Each raw input `x` is mapped to `φ(x) ∈ ℝ⁴`. All parameters are fixed constants, not fitted.

### 2.1 Monod kinetics — substrate utilisation

```
μ(S) = μ_max · S / (K_s + S)        S ≡ VS,  K_s = 15.0,  μ_max = 1.0
```

Microbial growth rate as a function of available substrate: linear when substrate is scarce,
saturating when abundant. Over the observed VS range (81.25–85 %) this is deep in the saturated
regime, returning values ≈ 0.844–0.850.

### 2.2 Arrhenius factor — temperature dependence

```
T_actual = 30.0 + (TS − 20.0) · 1.5          [°C]
k(T)     = exp[ −(E_a/R) · (1/T_K − 1/T_ref,K) ]
E_a = 50 000 J/mol,  R = 8.314 J/(mol·K),  T_ref = 25 °C
```

No temperature column exists in the dataset, so **`TS` is used as a proxy**, linearly mapped onto
a 30–45 °C mesophilic window. Over the observed TS range this yields factors of roughly 1.6–2.6.

> This is the most assumption-heavy component of the embedding. It encodes a *monotone increasing*
> rate with TS, which is a modelling choice rather than a measured relationship.

### 2.3 pH-activity window

```
f(pH) = exp[ −½ · ((pH − pH_opt) / σ_pH)² ]      pH_opt = 7.0,  σ_pH = 0.8
```

Gaussian activity window around the methanogenic optimum. The observed pH range (7.00–7.15) sits
almost exactly at the optimum, so this feature is near-constant at ≈ 0.99 across the dataset and
carries very little information.

### 2.4 Microbial lifecycle — modified Gompertz

```
growth(t)  = 1 − exp[ −α · max(0, t − t_lag) ]
decline(t) = exp[ −β · max(0, t − t_dec) ]
g(t)       = growth(t) · decline(t)

t_lag = 1.5,  t_peak = 7.0,  t_dec = 12.0,  α = 0.5,  β = 0.3
```

Four-phase digestion dynamics: lag → exponential → stationary → decline. Because the dataset stops
at day 10 and `t_dec = 12`, **the decline term is identically 1 over the observed range** — only
the growth sigmoid is active.

| Parameter | Value | Role |
|:---|:---:|:---|
| `K_s` | 15.0 | Monod half-saturation constant |
| `μ_max` | 1.0 | Maximum specific growth rate |
| `E_a` | 50 000 J/mol | Activation energy |
| `T_ref` | 25 °C | Arrhenius reference temperature |
| `pH_opt` | 7.0 | Methanogenic pH optimum |
| `σ_pH` | 0.8 | Width of pH activity window |
| `t_lag` | 1.5 d | End of lag phase |
| `t_dec` | 12.0 d | Onset of decline |
| `α` | 0.5 | Growth-phase rate |
| `β` | 0.3 | Decline-phase rate |

---

## 3. The kernel

`PhysicsInformedBioEnergyKernel` subclasses `sklearn.gaussian_process.kernels.Kernel`.

### 3.1 Form

```
K(x, x') = (1 − w) · σ²_data · exp( −‖x − x'‖² / 2ℓ² )
         +       w · σ²_phy  · exp( −‖φ(x) − φ(x')‖² / 2ℓ²_phy )
         +           σ²_noise · δ(x, x')
```

Both terms are squared-exponential kernels — one over the standardised raw inputs, one over the
physics embedding. A convex combination of two PSD kernels is PSD, so `K` is a valid covariance
function. The noise term is added only when `X` and `Y` are the same array.

### 3.2 Hyperparameters

| Name | Init | Bounds | Fitted |
|:---|:---:|:---:|:---:|
| `length_scale` (ℓ) | 1.0 | (1e-3, 1e3) | **2.217** |
| `physics_length_scale` (ℓ_phy) | 1.0 | (1e-3, 1e3) | **5.038** |
| `sigma_data` (σ_data) | 10.0 | (1e-3, 1e3) | **1.723** |
| `sigma_physics` (σ_phy) | 10.0 | (1e-3, 1e3) | **0.002** |
| `sigma_noise` (σ_noise) | 1.0 | (1e-5, 1e1) | — |
| `physics_weight` (w) | 0.4 | *fixed* | 0.40 |

`w` is **not** a learnable hyperparameter — it is a fixed architectural choice declaring a 60 % data
/ 40 % physics split. The actual balance is therefore decided by the fitted amplitudes, and
`σ_phy = 0.002` means the physics branch contributes `w · σ²_phy ≈ 1.6 × 10⁻⁶` — numerically
negligible. **The marginal likelihood chose to ignore the physics prior.**

### 3.3 Gradients

scikit-learn requires `∂K/∂θ` for marginal-likelihood optimisation. Because `φ(·)` involves
`np.maximum` clamps and a scaler inverse-transform, analytic gradients are awkward, so the kernel
computes **central finite differences** (`ε = 1e-5`) by cloning itself at `θ ± ε` per hyperparameter.

This is correct but costly: each gradient evaluation triggers `2 · n_dims` full kernel
constructions, each recomputing `φ` in a Python loop. It is the dominant runtime cost and the
first thing to replace — see the roadmap.

---

## 4. Training

```python
GaussianProcessRegressor(
    kernel=pigp_kernel,
    n_restarts_optimizer=5,
    alpha=1e-6,
    normalize_y=True,
    random_state=42,
)
```

- **5 restarts** of L-BFGS-B from random initialisations, guarding against local optima in the LML.
- `normalize_y=True` centres and scales the target, so the GP's zero-mean prior is appropriate.
- `alpha=1e-6` is a Cholesky jitter term, separate from the kernel's own `σ²_noise`.

Converged log marginal likelihood: **27.5638**.

### 4.1 Benchmark kernels

Five stock alternatives are fitted under identical conditions:
`RBF`, `Matérn(ν=1.5)`, `Matérn(ν=2.5)`, `RationalQuadratic`, and `RBF + Matérn` —
each as `ConstantKernel(10.0) * <kernel> + WhiteKernel(1.0)`.

### 4.2 Cross-validation

`KFold(n_splits=5, shuffle=True, random_state=42)` over all 40 samples, comparing PI-GPR, RBF GP
and Matérn GP. Note this inherits the same grouping weakness as the train/test split.

---

## 5. Genetic-algorithm optimisation

The trained GP becomes the objective function for a PyGAD search over feedstock composition.

### 5.1 Configuration

| Setting | Value |
|:---|:---|
| Generations | 100 |
| Population | 30 |
| Parents mating | 10 |
| Genes | 2 — `Food_pct`, `Weed_pct` |
| Gene space | `range(0, 101)` each |
| Mutation | 50 % of genes |
| Seed | 42 |

### 5.2 Fitness function

For a candidate `(Food%, Weed%)`:

1. **Constraint.** If `Food% + Weed% ≠ 100`, return `−1000`. The constraint is enforced by
   rejection rather than by reparameterisation, so a large share of the population is wasted on
   invalid pairs — a reparameterised single gene `Food% ∈ [0,100]` with `Weed% = 100 − Food%`
   would be strictly better.
2. **Reconstruct the state** via the mixing laws from §1.2, with `pH = 7.1` and `Day = 10` held fixed.
3. **Predict** `ŷ, σ` from the GP.
4. **Return the lower confidence bound** `ŷ − 0.5σ`.

The LCB is what makes this uncertainty-aware: a blend the model rates highly but is unsure about
is penalised relative to a slightly lower-yield blend it is confident in. Day 10 is fixed because
it is the peak of the observed production curve.

### 5.3 Landscape visualisation

A 101 × 10 grid over (`Food_pct`, `Day`) is evaluated to produce the yield and uncertainty
surfaces in `assets/optimization-landscape.png`, which show how predictive variance grows away
from the four sampled blends.

---

## 6. Known limitations

1. **Random rather than grouped splitting.** Rows within a blend share six of seven features.
   A random split leaks near-duplicates into training, inflating every metric reported here.
   Leave-one-blend-out CV is the correct protocol and is the top roadmap item.

2. **Scaler fitted before splitting.** Minor leakage of global mean/variance into the test fold.

3. **The physics branch is inert.** `σ_phy = 0.002`. Any claim that the physics prior *caused*
   the accuracy is unsupported by this run — the data-driven RBF branch does the work.

4. **Two features are near-constant.** The pH-activity factor varies by < 1 % across the dataset
   and the lifecycle decline term never activates, so `φ(x)` is effectively two-dimensional.

5. **Arrhenius uses a proxy.** TS→temperature is an assumed linear map with no measured backing.

6. **GA extrapolates.** The search spans `Food% ∈ [0, 100]`; observations only cover `[25, 100]`.

7. **Sample size.** 40 observations across 4 blends is small for a seven-feature model, and the
   8-sample test set makes the reported R² high-variance.
