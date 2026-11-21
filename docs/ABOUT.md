# About This Project

## One-line description

> A Gaussian Process whose covariance function encodes anaerobic-digestion biochemistry — Monod kinetics, Arrhenius temperature dependence, pH-activity windows and microbial lifecycle dynamics — paired with a genetic algorithm that searches the learned surface for the optimal organic-waste feedstock blend.

---

## GitHub "About" blurb

Paste into the repository's **About** field (⚙️ next to *About* on the repo home page):

```
Physics-informed Gaussian Process Regression for biogas yield prediction — a custom
scikit-learn kernel embedding Monod, Arrhenius, pH-activity and microbial-lifecycle
physics, with uncertainty-aware genetic-algorithm feedstock optimisation.
```

### Suggested topics

```
machine-learning  gaussian-processes  physics-informed-machine-learning
scientific-machine-learning  custom-kernel  scikit-learn  uncertainty-quantification
genetic-algorithm  pygad  bayesian-optimization  biogas  anaerobic-digestion
renewable-energy  bioenergy  regression  python  jupyter-notebook
```

### Website field

Leave blank, or link the published paper once it is available.

---

## The problem

Anaerobic digestion converts organic waste into methane-rich biogas. Yield depends on a
tangle of interacting variables — feedstock composition, solids content, pH, carbon-to-nitrogen
ratio and how long the digester has been running. Finding a good operating point empirically is
slow and expensive: **a single feedstock trial runs for weeks**, and the search space is continuous.

Purely data-driven models can interpolate such trials, but they have no notion of the
biochemistry underneath. They extrapolate badly, offer no principled measure of their own
confidence, and give an operator no basis for deciding which experiment to run next.

## The approach

Rather than adding a physics penalty to a loss function — the usual physics-informed recipe —
this project puts the physics in the **prior**, by writing it into the GP's kernel.

1. **A physics embedding.** Four textbook relations map a raw digester state to a
   four-dimensional vector `φ(x)`: Monod substrate kinetics on volatile solids, an Arrhenius
   rate factor on total solids, a Gaussian pH-activity window, and a modified-Gompertz
   microbial lifecycle curve over digestion day.

2. **A dual-path kernel.** `PhysicsInformedBioEnergyKernel` — a from-scratch subclass of
   `sklearn.gaussian_process.kernels.Kernel` — combines an RBF over the standardised raw
   features with an RBF over `φ(x)`, plus a white-noise term. Two states are similar when
   their measurements agree *and* when the biology they imply agrees. All five amplitude and
   length-scale hyperparameters are declared to scikit-learn and fitted by maximising the
   log marginal likelihood.

3. **Uncertainty-aware search.** A PyGAD genetic algorithm optimises the feedstock ratio
   against a **lower-confidence-bound** objective, `ŷ − 0.5σ`, so the recommendation favours
   blends the model is confident about rather than ones where it happens to be optimistic.

## What was found

On the 40-observation dataset the pipeline reaches **R² = 0.9992** (RMSE 0.81 cm³/day) on a
held-out test set, with **5-fold CV R² = 0.9988 ± 0.0011**, and recommends a **49 % food waste /
51 % weed biomass** blend yielding **164.57 cm³/day** (95 % CI 162.58–166.55).

It also produced an instructive negative result: the fitted physics amplitude collapsed to
`σ_physics = 0.002`, and all six benchmarked kernels — including a plain RBF — agree to five
decimal places. On a dataset this small and this smooth, the physics prior had nothing left to
contribute. That finding is documented rather than hidden; see the
[Limitations](../README.md#-limitations--what-the-numbers-really-say) section and
[`RESULTS.md`](RESULTS.md).

## Skills demonstrated

| Area | Evidence |
|:---|:---|
| **Custom ML internals** | A scikit-learn `Kernel` subclass written from scratch: hyperparameter declarations, `__call__` with gradient support, `diag`, `is_stationary` |
| **Scientific ML** | Domain equations translated into a feature map and a valid PSD covariance structure |
| **Bayesian methods** | GP regression, marginal-likelihood optimisation, calibrated predictive intervals |
| **Optimisation** | Evolutionary search under constraints, with an acquisition-style LCB objective |
| **Evaluation rigour** | Six-kernel benchmark, k-fold CV, residual and uncertainty diagnostics |
| **Scientific honesty** | Negative results and evaluation-protocol weaknesses stated explicitly |
| **Communication** | Layered documentation — README for breadth, `METHODOLOGY.md` and `RESULTS.md` for depth |

## Status

Research prototype supporting a paper under review. The notebook is the single source of
truth and runs end to end; the roadmap in the README tracks the work needed to turn it into
a reusable package.
