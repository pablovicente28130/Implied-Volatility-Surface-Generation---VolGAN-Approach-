# VolGAN: A Generative Model for Arbitrage-Free Implied Volatility Surfaces
## Exhaustive Analysis — Reference Document for Implementation

**Paper:** Vuletic, M. & Cont, R. (2024). Applied Mathematical Finance, 31:4, 203-238.
**DOI:** 10.1080/1350486X.2025.2471317
**Code:** https://github.com/milenavuletic/VolGAN/

---

## 1. CONTEXT AND RESEARCH QUESTION

### Problem
Option prices are quoted via implied volatilities (IV), obtained by inverting Black-Scholes. The IV surface sigma_t(m, tau) — parameterized by moneyness m = K/S_t and time-to-maturity tau = T - t — summarizes the full cross-section of option prices at date t. Modeling the *joint dynamics* of the IV surface and the underlying is critical for hedging and risk management, but parametric models struggle with the high dimensionality and nonlinear constraints involved.

### What the paper proposes
VolGAN is a **fully data-driven conditional GAN** that generates realistic one-day-ahead joint scenarios for:
- the log-return of the underlying asset R_t
- the increment of the log-implied volatility surface Delta_g_t(m, tau)

It is trained on time series of market-quoted IV surfaces (SPX) and produces scenarios that:
1. Capture the covariance structure of IV co-movements (level, skew, curvature factors)
2. Reproduce non-Gaussian, asymmetric distributions with heavy tails
3. Learn time-varying correlations (leverage effect)
4. Satisfy static arbitrage constraints (via scenario re-weighting)

### Key distinction from prior work
Unlike classical GANs using binary cross-entropy (BCE) loss only (Goodfellow et al. 2014), VolGAN uses a **custom loss function** with Sobolev smoothness penalties, combined with a **scenario re-weighting** scheme (Cont & Vuletic 2023) to enforce arbitrage-free outputs.

---

## 2. IMPLIED VOLATILITY SURFACE — BACKGROUND

### Definition
The IV sigma_t(m, tau) is defined implicitly by:
```
C_t(m, tau) = C_BS(S_t, K, tau, sigma_t(m, tau))
```
where C_BS is the Black-Scholes call price formula:
```
C_BS = S_t * N(d1) - K * exp(-r*tau) * N(d2)

d1 = (-ln(m) + tau*(r + sigma^2/2)) / (sigma * sqrt(tau))
d2 = (-ln(m) + tau*(r - sigma^2/2)) / (sigma * sqrt(tau))
```

### Static Arbitrage Constraints
Not every function sigma(m, tau) is admissible. Call/put option prices must satisfy three constraints:

1. **Calendar spread (no-arbitrage in tau):** d_tau C_BS(S, K, tau, sigma(m, tau)) >= 0
2. **Monotonicity in moneyness:** d_m C_BS(S, K, tau, sigma(m, tau)) <= 0
3. **Butterfly (convexity in moneyness):** d^2_m C_BS(S, K, tau, sigma(m, tau)) >= 0

These translate to nonlinear constraints on sigma, d_m(sigma), d^2_m(sigma), d_tau(sigma).

### Arbitrage Penalty
Define relative call prices:
```
c(m, tau) := (1/S) * C_BS(S, K, tau, sigma) = N(d1) - m*exp(-r*tau)*N(d2)
```

The **arbitrage penalty** Phi(sigma(m, tau)) is the sum of three penalty functions:

**p1 — Calendar spread violations:**
```
p1(sigma) = sum_{i,j} (tau_j * (c(m_i, tau_j) - c(m_i, tau_{j+1})) / (tau_{j+1} - tau_j))^+
```

**p2 — Call spread violations:**
```
p2(sigma) = sum_{i,j} ((c(m_{i+1}, tau_j) - c(m_i, tau_j)) / (m_{i+1} - m_i))^+
```

**p3 — Butterfly spread violations:**
```
p3(sigma) = sum_{i,j} ((c(m_i, tau_j) - c(m_{i-1}, tau_j))/(m_i - m_{i-1}) - (c(m_{i+1}, tau_j) - c(m_i, tau_j))/(m_{i+1} - m_i))^+
```

Static arbitrage-free condition: **Phi(sigma(m, tau)) = 0**.

### Empirical Properties of IV Dynamics
The paper lists the stylized facts that a good IV model must capture:
- Non-flat cross-section (smile/skew + term structure)
- High positive autocorrelation and mean-reversion of IVs
- PCA explains most variance with few factors: PC1 = level, PC2 = skew, PC3 = curvature
- Negative correlation between underlying returns and projections on level/skew PCs (leverage effect)
- Non-Gaussian, asymmetric increment distributions
- Time-varying instantaneous correlations

---

## 3. VOLGAN ARCHITECTURE — COMPLETE SPECIFICATION

### 3.1 Notation and State Variables

**Log-implied volatility surface:**
```
g_t(m, tau) = log(sigma_t(m, tau))
Delta_g_t(m, tau) = g_{t+Delta_t}(m, tau) - g_t(m, tau)
```
where Delta_t = 1/252 (one trading day).

**Log-return of underlying:**
```
R_t = log(S_{t+Delta_t} / S_t)
```

**One-month realized volatility:**
```
gamma_t = sqrt((252/21) * sum_{i=0}^{20} R_{t-i*Delta_t}^2)
```

**Condition/input vector:**
```
a_t = (R_{t-Delta_t}, R_{t-2*Delta_t}, gamma_{t-Delta_t}, g_t(m, tau))
```

This contains: 2 lagged returns, 1 lagged realized vol, and the current log-IV surface on the grid.

### 3.2 Generator G

**Input:** condition a_t and i.i.d. noise z_t ~ N(0, I_d), with d = 32.

**Output:** simulated return and log-IV increment:
```
G(a_t, z_t) = (R_hat_t(z), Delta_g_hat_t(m, tau)(z))
```

**Architecture: 3-layer feedforward dense neural network**
- Layer 1: H = 16 neurons
- Layer 2: 2H = 32 neurons
- Layer 3 (output): N_tau * N_m + 1 neurons (IV grid + 1 return)
- Activation 1 (layers 1-2): **softplus**
- Activation 2 (output layer): **affine (linear)**

The noise dimension d = 32, hidden size H = 16.

### 3.3 Discriminator D

**Input:** either generator output or real data (r, Delta_g), together with condition a_t.

**Output:** scalar in (0, 1) — estimated probability that input is real data.

**Architecture: 2-layer feedforward dense neural network**
- Layer 1: H = 16 neurons, activation = **softplus**
- Layer 2: 1 neuron, activation = **sigmoid**

The discriminator is deliberately simpler than the generator to keep the networks balanced.

### 3.4 Training Objective

#### Generator Loss J^(G)
```
J^(G)(theta_d, theta_g) = -(1/2) * E[log(D(a_t, G(a_t, z_t; theta_g); theta_d))]
                          + a_m * E[L_m(g_t(m, tau) + G(a_t, z_t; theta_g)|_2)]
                          + a_tau * E[L_tau(g_t(m, tau) + G(a_t, z_t; theta_g)|_2)]
```

Where:
- First term: BCE for the discriminator output (classical GAN adversarial loss)
- `G(a_t, z_t; theta_g)|_2 = Delta_g_hat_t(m, tau)(z_t)` is the simulated log-IV increment
- The simulated log-IV surface is: `g_hat_t(m, tau)(z_t) = g_t(m, tau) + Delta_g_hat_t(m, tau)(z_t)`

#### Smoothness Penalties (Sobolev semi-norms)
**L_m — smoothness in moneyness:**
```
L_m(g) = sum_{i,j} (g(m_{i+1}, tau_j) - g(m_i, tau_j))^2 / |m_{i+1} - m_i|^2
```
This approximates ||d_m(g)||^2_{L2}.

**L_tau — smoothness in maturity:**
```
L_tau(g) = sum_{i,j} (g(m_i, tau_{j+1}) - g(m_i, tau_j))^2 / |tau_{j+1} - tau_j|^2
```
This approximates ||d_tau(g)||^2_{L2}.

These penalties are applied to the **simulated log-implied volatility surfaces** (not to the increments alone, but to g_t + Delta_g_hat).

#### Discriminator Loss J^(D)
Standard conditional GAN BCE:
```
J^(D)(theta_d, theta_g) = -(1/2) * E[log(D(a_t, (R_t, Delta_g_t(m, tau)); theta_d))]
                          -(1/2) * E[log(1 - D(a_t, G(a_t, z_t; theta_g); theta_d))]
```

#### Hyperparameter Calibration (a_m, a_tau)
Chosen by **gradient norm matching**:
1. Train VolGAN for n_grad = 25 epochs with BCE only (classical GAN)
2. At each generator update, compute gradient norms of BCE, L_m, L_tau w.r.t. theta_g
3. Set a_m = mean ratio of (grad norm BCE / grad norm L_m)
4. Set a_tau = mean ratio of (grad norm BCE / grad norm L_tau)
5. Restart training from same initialization with full loss (Eq. 13)

This ensures all three loss terms contribute comparably during training.

### 3.5 Scenario Re-weighting (Arbitrage Enforcement)

**Key insight:** The raw generator outputs are NOT guaranteed to be arbitrage-free. Rather than hard-constraining the architecture, VolGAN applies a post-hoc re-weighting.

Let P_0 be the generator's output distribution. Define a tilted measure:
```
dP_beta / dP_0 (omega) = exp(-beta * Phi(sigma(m, tau; omega))) / Z(beta)
```
where:
- Phi is the arbitrage penalty (Eq. 2)
- Z(beta) is the normalization constant
- beta controls the strength of penalization

**Sampling via Weighted Monte Carlo:**
Given N samples (R_hat^i, sigma_hat^i) from the generator, each scenario gets weight:
```
w^i = exp(-beta * Phi(sigma_hat^i)) / sum_{j=1}^{N} exp(-beta * Phi(sigma_hat^j))
```

**Adaptive beta calibration:**
```
beta(t) = 500 / max{w_i(t)}
```
This is based on KL divergence between the weighted and uniform distributions (Cont & Vuletic 2023).

**Weighted expectations and quantiles:**
```
E_beta[X] = sum_{i=1}^{N} w^i * x^i

F^{-1}_{X,beta}(q) = x_{(k)}, where k = min{j : sum_{i=1}^{j} w_{(i)} >= q}
```

### 3.6 Training Details
- Optimizer: **RMSProp**
- Learning rate: **0.0001** (both networks)
- N = 10,000 raw samples from generator
- Mini-batch size: n_batch = 100
- Training epochs: n_epochs = 10,000
- Alternating direction: 1 discriminator update per generator update
- Expectation computed by sample averages (ergodicity assumption)

**Important note from the paper:** The arbitrage penalty Phi was NOT included directly in the generator loss. The authors state this made no notable difference — the smoothness penalty indirectly enforces shape constraints.

---

## 4. DATA AND PREPROCESSING

### Data Source
- **Options data:** OptionMetrics (Option Prices file), SPX options
- **VIX:** Historical VIX closing prices from CBOE
- **Risk-free rate:** Median rate implied by put-call parity from option mid-prices

### Time Period
- Full period: **3 Jan 2000 — 28 Feb 2023**
- Training: **3 Jan 2000 — 16 Jun 2018**
- Test: **17 Jun 2019 — 28 Feb 2023**

### Moneyness-Maturity Grid
```
m in {0.6, 0.7, 0.8, 0.9, 0.95, 1, 1.05, 1.1, 1.2, 1.3, 1.4}  (N_m = 11)
tau in {1/252, 1/52, 2/52, 1/12, 1/6, 1/4, 1/2, 3/4, 1}          (N_tau = 9)
```
Grid size: 11 x 9 = 99 IV points per surface.

Output dimension of generator: 99 (IV grid) + 1 (return) = **100**.

### IV Surface Smoothing
Vega-weighted Nadaraya-Watson kernel smoothing with 2D Gaussian kernel:
```
sigma_hat(m', tau') = [sum_{m,tau} kappa(m, tau) * k(m - m', tau - tau') * sigma(m, tau)] / [sum_{m,tau} kappa(m, tau) * k(m - m', tau - tau')]
```
where kappa(m, tau) is the vega and k is a 2D Gaussian kernel with bandwidths h1, h2.

**Bandwidth selection:** Grid search over h1, h2 in [0.002, 0.1] with step 0.002, minimizing the arbitrage penalty Phi on 100 randomly sampled days. Optimal: **(h1, h2) = (0.002, 0.046)**.

For general sigma_t(m, tau) off-grid: **bilinear interpolation** (first in moneyness, then in maturity). Extrapolation is linear.

---

## 5. OUT-OF-SAMPLE RESULTS

### 5.1 Discriminator as Anomaly Detector
The trained discriminator assigns low scores (< 0.2) to extreme market events: 2008 financial crisis (in-sample) and Covid-19 start (out-of-sample). This is an interesting by-product.

### 5.2 Smoothness and Arbitrage

| Metric | Market Data | BCE GAN | Raw VolGAN | VolGAN (reweighted) |
|--------|------------|---------|------------|---------------------|
| Mean arb. penalty | 0.0096 | 2.4635 | 0.0199 | 0.0127 |
| Std | 0.0628 | 0.9086 | 0.088 | 0.0620 |
| Median | 0.0005 | 2.3164 | 0.003 | 0.0014 |

BCE GAN produces highly irregular surfaces with massive arbitrage violations. VolGAN (with smoothness penalties) achieves arbitrage levels close to market data. Scenario re-weighting further reduces arbitrage.

### 5.3 Next-Day Forecasting (Exceedance Ratios)
For well-calibrated 95% confidence intervals, we expect ~2.5% of data below the 2.5% quantile and ~2.5% above the 97.5% quantile.

| Variable | 0.01 | 0.025 | 0.975 | 0.99 |
|----------|------|-------|-------|------|
| SPX return | 25.32% | 29.19% | 82.00% | 83.55% |
| 3-month ATM vol | 13.95% | 15.16% | 49.61% | 54.61% |
| 3-month OTM vol | 76.978% | 78.81% | 92.85% | 93.80% |
| 1-month ATM vol | 9.82% | 11.28% | 42.89% | 48.41% |
| 1-week ATM vol | 20.41% | 22.05% | 59.17% | 63.22% |
| 1-day ATM vol | 19.90% | 21.79% | 60.12% | 64.34% |
| VIX | 34.37% | 35.23% | 52.67% | 55.04% |

**Interpretation:** VolGAN underestimates extreme values (especially high implied vols and VIX). The CIs are too narrow for tails. Best overall forecasts are for the underlying return. The re-weighting (beta > 0) narrows CIs; without it (beta = 0), forecasts can actually improve (Table 3), especially for SPX returns, short-maturity ATM vol, and VIX.

### 5.4 Time-Varying Correlations (Leverage Effect)
VolGAN learns time-varying instantaneous correlations between returns and 1-month ATM vol increments. The correlation rho_t varies outside the 95% CI of a constant rho, rejecting H0 (constant correlation). This is a feature that parametric models with fixed correlation cannot capture.

### 5.5 Non-Gaussian Distributions
Simulated index returns and ATM vol increments exhibit:
- Asymmetry (negative skew for returns)
- Exponentially decaying heavy tails
- Both features match empirical distributions

### 5.6 Principal Component Analysis
Out-of-sample PCA on simulated log-IV increments:

| PC Rank | Data (variance explained) | VolGAN (variance explained) |
|---------|--------------------------|----------------------------|
| First | 51.25% | 45.31% +/- 1.84% |
| Second | 34.00% | 25.69% +/- 0.88% |
| Third | 5.01% | 12.76% +/- 0.55% |

- PC1 and PC2 significance is similar (level and skew)
- PC3 is more significant in VolGAN than in data (the model overweights curvature)
- Inner products of eigenvectors (data vs. VolGAN): PC1: 0.921, PC2: 0.921, PC3: 0.798 — strong alignment

### 5.7 Correlation Structure
Average Pearson correlations (VolGAN blue vs. data red, out-of-sample):

| | Delta log S | Delta X^1 | Delta log sigma^ATM | Delta log v |
|---|---|---|---|---|
| Delta log S | 1.00 | -0.84 / -0.76 | -0.86 / -0.77 | -0.55 / -0.71 |
| Delta X^1 | | 1.00 | 0.95 / 0.89 | 0.66 / 0.84 |
| Delta log sigma^ATM | | | 1.00 | 0.72 / 0.96 |
| Delta log v | | | | 1.00 |

(Format: VolGAN / Data)

VolGAN reproduces the correct signs and approximate magnitudes. VIX correlations are slightly lower in VolGAN than in data.

---

## 6. APPLICATION TO HEDGING

### 6.1 Setup
Portfolio: long straddle (1-month call + put at K = 1.2 * S_0), hedged until expiry, then rolled.

### 6.2 Hedging Methods Compared

1. **BS delta hedge:** Hedge with underlying only, using BS delta.
2. **BS delta-vega hedge:** Hedge with underlying + ATM call, using BS greeks.
3. **VolGAN + ATM:** Regression-based hedge using VolGAN scenarios, with underlying + ATM call at t=0.
4. **VolGAN + LASSO:** Regression-based hedge with automatic instrument selection via LASSO from a pool of calls and puts.
5. **VolGAN + LASSO + Scenario Weighting:** Same as 4, but with re-weighted scenarios.

### 6.3 Scenario-Based Regression Hedging
Interpret the one-step portfolio evolution as a regression:
```
Delta V_t = V_{t+Delta_t}(omega_j) - V_t = A_t + sum_i phi^i_t * (H^i(omega_j)_{t+Delta_t} - H^i_t) + epsilon_j
```
where omega_j are VolGAN scenarios. Hedge ratios phi^i_t are OLS coefficients from regressing simulated portfolio changes on simulated hedging instrument changes.

### 6.4 LASSO Instrument Selection
Pool of hedging instruments (same expiry as straddle):
- Puts: strikes 0.9*S_0, 0.95*S_0, 0.975*S_0
- Calls: strikes S_0, 1.025*S_0, 1.05*S_0, 1.1*S_0

L1 regularization parameter alpha searched in [0, 1] with step 0.1, calibrated by in-sample R^2 and MSE at position entry. LASSO used for instrument selection only at t=0; subsequent hedge ratios via OLS.

### 6.5 Key Finding: Scenario Re-weighting Hurts Hedging
Raw VolGAN outputs (beta = 0) produce better hedges than re-weighted outputs. This is because raw outputs better mimic the market (including small arbitrage), while re-weighting concentrates weight on few scenarios, narrowing the effective sample.

### 6.6 Hedging Results

| Statistic | VolGAN+LASSO | VolGAN+ATM | BS delta | BS delta-vega |
|-----------|-------------|------------|----------|---------------|
| Mean tracking error | **-0.051** | 0.056 | -0.614 | 1.541 |
| Std dev | 5.766 | **4.940** | 6.755 | 28.307 |
| 5% VaR | **5.815** | 7.314 | 8.310 | 7.258 |
| 2.5% VaR | **8.095** | 10.692 | 13.300 | 10.701 |
| 1% VaR | **13.172** | 13.730 | 34.023 | 14.181 |

**Key results:**
- VolGAN+LASSO has the best mean (closest to zero) and best VaR at all levels
- VolGAN+ATM has the lowest standard deviation
- BS delta-vega is **unstable** — it shows extreme instability during Covid-19 (the vega-based hedge ratio explodes when vega becomes small)
- All VolGAN methods outperform BS benchmarks
- LASSO typically selects 2-3 options during turbulence, 0 during calm periods — consistent with PCA showing 2-3 significant factors

### 6.7 Number of Hedging Instruments
LASSO selects hedging instruments adaptively:
- Calm periods: 0 options (underlying only)
- Turbulent periods (Covid, Ukraine war): 2-3 options
- Very rare: up to 7 options (extreme events with no regularization)

This is consistent with the PCA result of 3 dominant factors in IV dynamics.

---

## 7. CRITICAL ASSESSMENT

### Strengths
1. **Clean architecture:** Simple feedforward nets (16-32 neurons) — no recurrence, no convolutions. Reproducible.
2. **Smoothness penalty is elegant:** Sobolev semi-norms avoid the irregular surfaces of BCE GANs while keeping the architecture simple.
3. **Scenario re-weighting is a principled approach** to arbitrage enforcement without hard architectural constraints.
4. **Comprehensive validation:** PCA, correlation structure, distributional properties, hedging application — the paper covers many angles.
5. **Practical hedging application** with LASSO instrument selection is genuinely useful.
6. **Code available** on GitHub.

### Weaknesses and Limitations
1. **Exceedance ratios are poor for tails.** The model systematically underestimates extreme implied volatility levels. For risk management (VaR, stress testing), this is a significant limitation.
2. **Re-weighting paradox:** The arbitrage re-weighting (which is a core selling point) actually *hurts* hedging performance. The paper acknowledges this but the implication is that in practice you may not want to use re-weighting for the main application (hedging).
3. **PC3 overweighted:** VolGAN explains 12.76% of variance with PC3 vs. 5.01% in data. This suggests the model does not perfectly learn the relative importance of curvature dynamics.
4. **No recalibration tested:** The model is trained once and evaluated 4.5 years out-of-sample. While stability is demonstrated, no rolling/expanding window re-training is explored — a standard practice in applied finance.
5. **Single asset:** Only SPX is tested. Generalization to other underlyings, asset classes, or less liquid markets is untested.
6. **Small network:** 16-32 neurons may be limiting. No ablation on architecture size is provided.
7. **Condition vector is minimal:** Only 2 lagged returns + 1 realized vol + current surface. No macro variables, no order flow, no term structure of interest rates. This is deliberate (data-driven philosophy) but limits the conditioning information.
8. **The smoothing step is critical but somewhat opaque:** The choice of Nadaraya-Watson kernel with specific bandwidths strongly affects the input quality. The paper treats this as preprocessing, but errors here propagate through the entire model.

---

## 8. IMPLEMENTATION REFERENCE — KEY PARAMETERS SUMMARY

| Parameter | Value | Notes |
|-----------|-------|-------|
| Noise dimension d | 32 | i.i.d. N(0, I_32) |
| Generator hidden size H | 16 | Layer 1: 16, Layer 2: 32 |
| Generator layers | 3 | Dense feedforward |
| Generator activations | softplus (hidden), affine (output) | |
| Discriminator hidden size | 16 | Single hidden layer |
| Discriminator layers | 2 | Dense feedforward |
| Discriminator activations | softplus (hidden), sigmoid (output) | |
| Optimizer | RMSProp | Both networks |
| Learning rate | 0.0001 | Both networks |
| N (samples from generator) | 10,000 | Per evaluation |
| Mini-batch size | 100 | |
| n_grad (gradient matching epochs) | 25 | Phase 1: BCE only |
| n_epochs (main training) | 10,000 | Phase 2: full loss |
| Training alternation | 1:1 | 1 D update per G update |
| Moneyness grid | {0.6, 0.7, 0.8, 0.9, 0.95, 1, 1.05, 1.1, 1.2, 1.3, 1.4} | 11 points |
| Maturity grid | {1/252, 1/52, 2/52, 1/12, 1/6, 1/4, 1/2, 3/4, 1} | 9 points |
| Output dimension | 100 | 99 IV grid + 1 return |
| Condition dimension | 2 + 1 + 99 = 102 | 2 returns + 1 realized vol + 99 IV |
| Input to generator | 102 (condition) + 32 (noise) = 134 | |
| beta(t) for re-weighting | 500 / max{w_i(t)} | Adaptive |
| Kernel bandwidths (h1, h2) | (0.002, 0.046) | For IV smoothing |
| Realized vol window | 21 days | ~1 month |
| Delta_t | 1/252 | Daily |

---

## 9. LOSS FUNCTIONS — COMPLETE SPECIFICATION FOR CODING

### Generator Loss (to MINIMIZE)
```python
# Pseudocode
def generator_loss(D, G, a_t, z_t, g_t, alpha_m, alpha_tau):
    # Generate fake samples
    R_hat, Delta_g_hat = G(a_t, z_t)

    # Simulated log-IV surface
    g_hat = g_t + Delta_g_hat  # shape: (batch, N_m, N_tau)

    # BCE term (want D to classify fakes as real)
    fake_input = concat(R_hat, Delta_g_hat)
    bce_term = -0.5 * mean(log(D(a_t, fake_input)))

    # Smoothness in moneyness
    L_m = sum over (i,j) of (g_hat[m_{i+1}, tau_j] - g_hat[m_i, tau_j])^2 / (m_{i+1} - m_i)^2

    # Smoothness in maturity
    L_tau = sum over (i,j) of (g_hat[m_i, tau_{j+1}] - g_hat[m_i, tau_j])^2 / (tau_{j+1} - tau_j)^2

    loss = bce_term + alpha_m * mean(L_m) + alpha_tau * mean(L_tau)
    return loss
```

### Discriminator Loss (to MINIMIZE)
```python
def discriminator_loss(D, G, a_t, z_t, R_real, Delta_g_real):
    # Real data
    real_input = concat(R_real, Delta_g_real)
    loss_real = -0.5 * mean(log(D(a_t, real_input)))

    # Fake data
    R_hat, Delta_g_hat = G(a_t, z_t)
    fake_input = concat(R_hat, Delta_g_hat)
    loss_fake = -0.5 * mean(log(1 - D(a_t, fake_input)))

    loss = loss_real + loss_fake
    return loss
```

### Scenario Re-weighting
```python
def compute_weights(generated_surfaces, beta):
    # generated_surfaces: (N, N_m, N_tau) IV surfaces (NOT log)
    # Compute arbitrage penalty for each
    penalties = [arbitrage_penalty(surf) for surf in generated_surfaces]

    log_weights = [-beta * p for p in penalties]
    # Softmax for numerical stability
    max_lw = max(log_weights)
    weights = [exp(lw - max_lw) for lw in log_weights]
    total = sum(weights)
    weights = [w / total for w in weights]
    return weights

def adaptive_beta(raw_weights_from_generator):
    # raw_weights are uniform (1/N) initially
    # After first pass, use:
    return 500.0 / max(raw_weights_from_generator)
```

---

## 10. DATA PIPELINE SUMMARY

1. **Raw option prices** from OptionMetrics -> compute IV by inverting BS
2. **Risk-free rate** from put-call parity median
3. **Kernel smoothing** of raw IV onto (m, tau) grid using Nadaraya-Watson with vega weighting
4. **Log-transform**: g_t = log(sigma_t)
5. **Compute increments**: Delta_g_t = g_{t+1} - g_t
6. **Compute returns**: R_t = log(S_{t+1}/S_t)
7. **Compute realized vol**: gamma_t from 21-day rolling window
8. **Assemble condition vectors**: a_t = (R_{t-1}, R_{t-2}, gamma_{t-1}, g_t)
9. **Train VolGAN** on (a_t, R_t, Delta_g_t) pairs

---

## 11. GITHUB REPOSITORY

Code: https://github.com/milenavuletic/VolGAN/

This should be consulted for implementation details not fully specified in the paper (e.g., exact initialization, data loading pipeline, interpolation details).
