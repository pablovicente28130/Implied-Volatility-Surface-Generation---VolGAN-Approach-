# VolGAN — Generative Model for Arbitrage-Free Implied Volatility Surfaces

Implementation of **VolGAN**, a conditional Generative Adversarial Network (GAN) that models the joint dynamics of implied volatility (IV) surfaces and underlying asset returns, ensuring arbitrage-free properties.

> Based on: *"VolGAN: A Generative Model for Arbitrage-Free Implied Volatility Surfaces"*  
> Vuletić & Cont (2024), *Applied Mathematical Finance*, 31(4), 203–238  
> DOI: [10.1080/1350486X.2025.2471317](https://doi.org/10.1080/1350486X.2025.2471317)

---

## Overview

Implied volatility surfaces are central objects in financial derivatives pricing. This project implements a data-driven conditional GAN that:

- Generates realistic joint scenarios of **(return, IV surface increment)** pairs
- Conditions generation on the current market state (lagged returns, realized volatility, current IV surface)
- Enforces **static arbitrage-free** constraints via Sobolev smoothness penalties and scenario re-weighting
- Reproduces key stylized facts: leverage effect, heavy tails, PCA structure (level, skew, curvature factors)

Two pipelines are provided:
- `Projet_VOLGAN.py` — synthetic data pipeline (Heston-style dynamics)
- `VolGAN_ORATS.ipynb` — real market data pipeline (SPY options via ORATS)

---

## Architecture

### Grid
- **Moneyness:** m ∈ {0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3} (7 points)
- **Maturities:** τ ∈ {1/52, 1/12, 1/4, 1/2, 1.0} years (5 points)
- **Surface size:** 7 × 5 = 35 points

### State / Condition vector (dim = 38)
| Variable | Dimension | Description |
|----------|-----------|-------------|
| R_{t-1}, R_{t-2} | 2 | Lagged log-returns |
| γ_{t-1} | 1 | 21-day realized volatility |
| g_t | 35 | Current log-IV surface |

### Generator (MLP, ~3,400 params)
```
Input: z (32) + condition (38) = 70
→ Linear(70, 16) + Softplus
→ Linear(16, 32) + Softplus
→ Linear(32, 36)   # output: (R̂_t, Δĝ_t)
```

### Discriminator (MLP, ~1,500 params)
```
Input: condition (38) + output (36) = 74
→ Linear(74, 16) + Softplus
→ Linear(16, 1)    # logit (no sigmoid)
```

---

## Loss Functions

### Generator
```
J_G = L_BCE + α_m · L_m + α_τ · L_τ
```
- **L_BCE**: adversarial cross-entropy
- **L_m, L_τ**: Sobolev smoothness penalties in moneyness and maturity dimensions

### Discriminator
```
J_D = 0.5 · E[log D(real)] + 0.5 · E[log(1 − D(fake))]
```

---

## Training

Two-phase training procedure:

### Phase 1 — Gradient Norm Matching (GNM, 25 epochs)
Calibrates smoothness weights:
```
α = mean(‖∇L_BCE‖ / ‖∇L_penalty‖)
```

### Phase 2 — Full VolGAN Training (up to 10,000 epochs)
- Optimizer: RMSProp (lr = 1e-4)
- Batch size: 100
- 1 discriminator update per generator update
- Mode collapse detection every 500 epochs (variance-based early stopping)

### Arbitrage Enforcement (post-hoc)
Scenario re-weighting with adaptive β:
```
w_i = exp(−β · Φ_i) / Σ exp(−β · Φ_j)
β_adaptive = 500 / max{w_i}
```
where Φ(σ) penalizes calendar spread, call spread, and butterfly spread violations.

---

## Results (synthetic data, 10,000 epochs)

| Metric | Data | VolGAN |
|--------|------|--------|
| PCA Variance PC1 | 68.87% | 74.85% |
| PCA Variance PC2 | 11.08% | 11.64% |
| PCA Inner Product PC1 | — | 0.980 |
| PCA Inner Product PC2 | — | 0.975 |
| PCA Inner Product PC3 | — | 0.829 |
| Arbitrage Penalty Φ | 0.000256 | 0.003976 |
| ATM 95% Coverage | — | 80.74% |

---

## Project Structure

```
projet_ML_vf/
├── Projet_VOLGAN.py          # Main implementation (synthetic data pipeline)
├── VolGAN_ORATS.ipynb        # Real market data pipeline (ORATS / SPY)
├── VolGAN_paper_analysis.md  # Detailed paper analysis and notes
├── generator.pt              # Trained generator weights
├── discriminator.pt          # Trained discriminator weights
├── normalizer.pkl            # Fitted data normalizer
├── history.json              # Training loss history (10,000 epochs)
└── metrics.json              # Out-of-sample evaluation metrics
```

---

## Requirements

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib jupyter
```

| Library | Purpose |
|---------|---------|
| PyTorch | Generator, Discriminator, training loop |
| NumPy / Pandas | Data manipulation |
| SciPy | Grid interpolation (ORATS pipeline) |
| scikit-learn | PCA evaluation |
| Matplotlib | Visualization |

---

## Usage

### Run full training (synthetic data)
```bash
python Projet_VOLGAN.py
```

This will:
1. Generate synthetic IV surfaces (Heston-style, 3,000 days)
2. Run Phase 1 (gradient norm matching) to calibrate α_m, α_τ
3. Run Phase 2 (full VolGAN training, 10,000 epochs)
4. Save `generator.pt`, `discriminator.pt`, `normalizer.pkl`
5. Evaluate and save `metrics.json` and `history.json`

### Load a pre-trained model
```python
import torch
from Projet_VOLGAN import Generator, Normalizer
import pickle

generator = Generator(latent_dim=32, condition_dim=38, output_dim=36)
generator.load_state_dict(torch.load("generator.pt"))

with open("normalizer.pkl", "rb") as f:
    normalizer = pickle.load(f)
```

### Real market data (ORATS)
Open and run `VolGAN_ORATS.ipynb` in Jupyter. Requires access to ORATS option data for SPY.

---

## Key Design Choices

- **Softplus activations** (not ReLU/tanh) for smooth gradient flow — important for Sobolev penalties
- **No hard arbitrage constraints** in architecture; enforced post-hoc via re-weighting
- **Chronological train/test split** to avoid lookahead bias
- **RMSProp** preferred over Adam for GAN stability
- **Logit output** (no sigmoid) in discriminator for numerical stability

---

## Limitations

- Tail underestimation: model is conservative on extreme volatility moves
- PC3 slightly overweighted (12.76% vs. 5.01% in data)
- ATM 95% coverage of 80.74% (below the 95% target)
- Single-asset, no rolling recalibration
- ORATS notebook limited to ~5 years of data vs. 23 years in the original paper

---

## References

- Vuletić, M., & Cont, R. (2024). *VolGAN: A Generative Model for Arbitrage-Free Implied Volatility Surfaces*. Applied Mathematical Finance, 31(4), 203–238.
- Black, F., & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*.
- Goodfellow, I., et al. (2014). *Generative Adversarial Nets*. NeurIPS.
