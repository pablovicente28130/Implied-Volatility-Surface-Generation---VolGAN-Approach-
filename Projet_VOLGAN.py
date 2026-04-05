"""
VolGAN — Generative Model for Arbitrage-Free Implied Volatility Surfaces - Vuletić & Cont (2024) 

Convention d'indexation de la grille (invariante) :
  g_flat[b, i*N_TAU + j]  ←→  g[b, i, j]
  axe i = moneyness (0..N_M-1), axe j = maturité (0..N_TAU-1)
  La maturité varie le plus vite (C-order).

Usage :
  - Grille par défaut : 7 × 5 = 35 pts (données synthétiques, cf. main()).
  - Pour une grille personnalisée (ex. ORATS 9 × 5 = 45 pts), appeler
    configure_grid(moneyness, maturities) AVANT toute instanciation
    de Generator / Discriminator.
"""

import copy
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset

_NORMAL01 = torch.distributions.Normal(0., 1.)


# ─────────────────────────────────────────────────────────────────────────────
# 0.  REPRODUCTIBILITÉ
# ─────────────────────────────────────────────────────────────────────────────

def set_seeds(seed: int = 42) -> None:
    """Seeds globales pour reproductibilité totale."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    # Algorithmes déterministes PyTorch (certaines opérations peuvent ne pas être supportées)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 1.  GRILLE (m, τ)
# ─────────────────────────────────────────────────────────────────────────────

# Grille 7 × 5 = 35 points
MONEYNESS  = torch.tensor([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])   # N_m = 7
MATURITIES = torch.tensor([1/52, 1/12, 1/4, 1/2, 1.0])            # N_τ = 5
N_M, N_TAU  = len(MONEYNESS), len(MATURITIES)
SURFACE_DIM = N_M * N_TAU   # 35

DIM_A = 2 + 1 + SURFACE_DIM    # 38  (R_{t-1}, R_{t-2}, γ_{t-1}, g_t)
DIM_X = 1 + SURFACE_DIM        # 36  (R_t, Δg_t)

# Indices ATM : m=1.0 (i=3), τ=1/4 (j=2)
ATM_M_IDX   = 3
ATM_TAU_IDX = 2
ATM_IDX     = ATM_M_IDX * N_TAU + ATM_TAU_IDX   # 17


def configure_grid(
    moneyness,
    maturities,
    atm_m_idx: int = None,
    atm_tau_idx: int = None,
) -> None:
    """
    Reconfigure la grille (m, τ) du module AVANT l'instanciation des modèles.

    Paramètres
    ----------
    moneyness  : vecteur de moneyness (ex. [0.7, 0.8, ..., 1.3])
    maturities : vecteur de maturités en années
    atm_m_idx  : indice ATM en moneyness (défaut : indice le plus proche de 1.0)
    atm_tau_idx: indice ATM en maturité  (défaut : indice le plus proche de 0.25)
    """
    global MONEYNESS, MATURITIES, N_M, N_TAU, SURFACE_DIM, DIM_A, DIM_X
    global ATM_M_IDX, ATM_TAU_IDX, ATM_IDX

    MONEYNESS  = torch.tensor(moneyness, dtype=torch.float32)
    MATURITIES = torch.tensor(maturities, dtype=torch.float32)
    N_M        = len(MONEYNESS)
    N_TAU      = len(MATURITIES)
    SURFACE_DIM = N_M * N_TAU
    DIM_A       = 2 + 1 + SURFACE_DIM
    DIM_X       = 1 + SURFACE_DIM

    if atm_m_idx is None:
        atm_m_idx = int(torch.argmin(torch.abs(MONEYNESS - 1.0)).item())
    if atm_tau_idx is None:
        atm_tau_idx = int(torch.argmin(torch.abs(MATURITIES - 0.25)).item())

    ATM_M_IDX   = atm_m_idx
    ATM_TAU_IDX = atm_tau_idx
    ATM_IDX     = ATM_M_IDX * N_TAU + ATM_TAU_IDX

    print(f"[configure_grid] {N_M}×{N_TAU} = {SURFACE_DIM} pts  "
          f"| DIM_A={DIM_A}, DIM_X={DIM_X}  "
          f"| ATM=(m={MONEYNESS[ATM_M_IDX]:.2f}, τ={MATURITIES[ATM_TAU_IDX]:.4f})")


def _grid_index(i_m: int, j_tau: int) -> int:
    """Indice plat pour la case (moneyness i_m, maturité j_tau)."""
    return i_m * N_TAU + j_tau


def _test_grid_indexing() -> None:
    """ Vérifie la bijection flat ↔ 3D pour toutes les 35 cases."""
    g_flat = torch.arange(SURFACE_DIM, dtype=torch.float32).unsqueeze(0)
    g_3d   = g_flat.view(1, N_M, N_TAU)
    for i in range(N_M):
        for j in range(N_TAU):
            assert g_flat[0, _grid_index(i, j)] == g_3d[0, i, j], \
                f"Incohérence grille : ({i},{j}) → flat={_grid_index(i,j)}"
    assert _grid_index(ATM_M_IDX, ATM_TAU_IDX) == ATM_IDX, \
        f"ATM_IDX={ATM_IDX} incohérent avec la grille"
    print(f"[TEST] Indexation grille OK  |  ATM=(m={MONEYNESS[ATM_M_IDX]:.1f},"
          f" τ={MATURITIES[ATM_TAU_IDX]:.4f}) → flat[{ATM_IDX}]")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  NORMALIZER
# ─────────────────────────────────────────────────────────────────────────────

class Normalizer:
    """
    Standardise condition a_t et cible x_t.
    Stats calculées sur le train set uniquement.

    Méthodes de dénormalisation factorisées  :
      inverse_g(g_norm)   → g en espace log-vol original
      inverse_dg(dg_norm) → Δg en espace original
      inverse_Rt(Rt_norm) → retour en espace original
      inverse_target(x_norm) → (R_t, Δg_t) en espace original  [compatibilité]
    """

    def __init__(self):
        self.R_mean   = 0.0;  self.R_std   = 1.0
        self.gam_mean = 0.0;  self.gam_std = 1.0
        self.g_mean:  np.ndarray = None
        self.g_std:   np.ndarray = None
        self.Rt_mean  = 0.0;  self.Rt_std  = 1.0
        self.dg_mean: np.ndarray = None
        self.dg_std:  np.ndarray = None

    def fit(self, conditions: np.ndarray, targets: np.ndarray) -> None:
        # R_mean/R_std : par colonne (R_{t-1}, R_{t-2}) — même distribution, cohérent avec g_mean/g_std
        self.R_mean   = conditions[:, :2].mean(axis=0)
        self.R_std   = conditions[:, :2].std(axis=0) + 1e-8
        self.gam_mean, self.gam_std = conditions[:, 2].mean(),  conditions[:, 2].std()   + 1e-8
        self.g_mean  = conditions[:, 3:].mean(axis=0)
        self.g_std   = conditions[:, 3:].std(axis=0)  + 1e-8
        self.Rt_mean,  self.Rt_std  = targets[:, 0].mean(),     targets[:, 0].std()      + 1e-8
        self.dg_mean = targets[:, 1:].mean(axis=0)
        self.dg_std  = targets[:, 1:].std(axis=0)  + 1e-8
        # Cache des tenseurs (évite _to_t à chaque forward)
        self._dg_std_t  = torch.tensor(self.dg_std,  dtype=torch.float32)
        self._dg_mean_t = torch.tensor(self.dg_mean, dtype=torch.float32)
        self._g_std_t   = torch.tensor(self.g_std,   dtype=torch.float32)
        self._g_mean_t  = torch.tensor(self.g_mean,  dtype=torch.float32)

    # ── Forward transforms (numpy) ────────────────────────────────────────

    def transform_condition(self, cond: np.ndarray) -> np.ndarray:
        out = cond.copy()
        out[:, :2] = (cond[:, :2] - self.R_mean)   / self.R_std
        out[:, 2]  = (cond[:, 2]  - self.gam_mean) / self.gam_std
        out[:, 3:] = (cond[:, 3:] - self.g_mean)   / self.g_std
        return out

    def transform_target(self, tgt: np.ndarray) -> np.ndarray:
        out = tgt.copy()
        out[:, 0]  = (tgt[:, 0]  - self.Rt_mean) / self.Rt_std
        out[:, 1:] = (tgt[:, 1:] - self.dg_mean) / self.dg_std
        return out

    # ── Inverse transforms (torch) — factorisés  ───────────────────────

    def _to_t(self, arr: np.ndarray, ref: torch.Tensor) -> torch.Tensor:
        """Convertit un array numpy en tenseur sur le bon device/dtype."""
        return torch.tensor(arr, dtype=ref.dtype, device=ref.device)

    def inverse_g(self, g_norm: torch.Tensor) -> torch.Tensor:
        """ g_norm (B, 35) → g en log-vol original."""
        std  = self._g_std_t.to(g_norm.device)
        mean = self._g_mean_t.to(g_norm.device)
        return g_norm * std + mean

    def inverse_dg(self, dg_norm: torch.Tensor) -> torch.Tensor:
        """ dg_norm (B, 35) → Δg en espace original."""
        std  = self._dg_std_t.to(dg_norm.device)
        mean = self._dg_mean_t.to(dg_norm.device)
        return dg_norm * std + mean

    def inverse_Rt(self, Rt_norm: torch.Tensor) -> torch.Tensor:
        """ Rt_norm (B,) ou (B,1) → retour en espace original."""
        return Rt_norm * self.Rt_std + self.Rt_mean

    def inverse_target(self, x_norm: torch.Tensor) -> torch.Tensor:
        """ x_norm (B, DIM_X) → (R_t, Δg_t) en espace original."""
        out = x_norm.clone()
        out[:, 0]  = self.inverse_Rt(x_norm[:, 0])
        out[:, 1:] = self.inverse_dg(x_norm[:, 1:])
        return out

    def get_surface_from_condition(self, cond_norm: torch.Tensor) -> torch.Tensor:
        """Extrait et dénormalise g_t depuis la condition normalisée."""
        return self.inverse_g(cond_norm[:, 3:])


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GÉNÉRATEUR  G : (z, a_norm) → (R̂_norm, Δĝ_norm)
# ─────────────────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    MLP 3 couches, softplus, couche finale affine.
    Travaille entièrement en espace normalisé.

    Couche 1 : (latent_dim + DIM_A) → H      + softplus
    Couche 2 : H                    → 2H     + softplus
    Couche 3 : 2H                   → DIM_X  (affine)
    """

    def __init__(self, latent_dim: int = 32, hidden: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + DIM_A, hidden),
            nn.Softplus(),
            nn.Linear(hidden, 2 * hidden),
            nn.Softplus(),
            nn.Linear(2 * hidden, DIM_X),
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, cond], dim=-1))

    def sample(self, cond: torch.Tensor) -> torch.Tensor:
        z = torch.randn(cond.shape[0], self.latent_dim, device=cond.device)
        return self.forward(z, cond)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DISCRIMINATEUR  D : (a_norm, x_norm) → logit ∈ ℝ
# ─────────────────────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    MLP 2 couches, softplus.
     Pas de Sigmoid — utiliser BCEWithLogitsLoss.

    Couche 1 : (DIM_A + DIM_X) → H   + softplus
    Couche 2 : H               → 1   (logit brut)
    """

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM_A + DIM_X, hidden),
            nn.Softplus(),
            nn.Linear(hidden, 1),
        )

    def forward(self, cond: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([cond, x], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PÉNALITÉS DE RÉGULARITÉ / Smoothness à la place de BCE  (semi-normes de Sobolev discrètes)
# ─────────────────────────────────────────────────────────────────────────────

def sobolev_penalties(
    g_flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    L_m(g) ≈ ||∂_m g||²_{L²}  et  L_τ(g) ≈ ||∂_τ g||²_{L²}
    Eqs. (11)–(12) du papier.

    g_flat : (B, SURFACE_DIM) — log-vol EN ESPACE ORIGINAL
    Convention : g_flat[b, i*N_TAU+j] ↔ g[b, i, j]
    returns : (scalaire, scalaire) — moyennes sur le batch
    """
    B = g_flat.shape[0]
    g = g_flat.view(B, N_M, N_TAU)

    dm   = (MONEYNESS[1:] - MONEYNESS[:-1]).to(g.device)      # (6,)
    Lm   = ((g[:, 1:, :] - g[:, :-1, :])**2
            / dm.view(1, -1, 1)**2).sum(dim=(1, 2)).mean()

    dtau = (MATURITIES[1:] - MATURITIES[:-1]).to(g.device)    # (4,)
    Ltau = ((g[:, :, 1:] - g[:, :, :-1])**2
            / dtau.view(1, 1, -1)**2).sum(dim=(1, 2)).mean()

    return Lm, Ltau


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PÉNALITÉ D'ARBITRAGE  Λ(σ)
# ─────────────────────────────────────────────────────────────────────────────

# NOTE : r=0.0 est une approximation conservative. Sur les données ORATS
# post-2022 (taux remontants), cela peut légèrement surestimer les prix
# de call relatifs pour les longues maturités. L'impact sur la pénalité
# d'arbitrage reste faible car p1, p2, p3 mesurent des différences de prix.
def _relative_call_price(sigma_flat: torch.Tensor, r: float = 0.0) -> torch.Tensor:
    """
    c(m,τ) = N(d₁) − m·e^{−rτ}·N(d₂)
    sigma_flat : (B, SURFACE_DIM)  — vol implicite > 0
    returns    : (B, N_M, N_TAU)
    """
    B     = sigma_flat.shape[0]
    s     = sigma_flat.view(B, N_M, N_TAU).clamp(min=1e-6)
    m     = MONEYNESS.to(s.device).view(1, N_M, 1)
    tau   = MATURITIES.to(s.device).view(1, 1, N_TAU)
    d1    = (-torch.log(m) + tau * (r + s**2 / 2)) / (s * tau.sqrt())
    d2    = d1 - s * tau.sqrt()
    N01   = _NORMAL01
    return N01.cdf(d1) - m * torch.exp(-torch.tensor(r, device=s.device) * tau) * N01.cdf(d2)


def arbitrage_penalty(sigma_flat: torch.Tensor) -> torch.Tensor:
    """
    Proxy discret des violations d'arbitrage statique.
    Λ(σ) = p₁(calendar) + p₂(call spread) + p₃(butterfly)
    Eqs. (2)–(5) du papier.

    sigma_flat : (B, SURFACE_DIM)
    returns    : (B,)
    """
    c   = _relative_call_price(sigma_flat)
    m   = MONEYNESS.to(sigma_flat.device)
    tau = MATURITIES.to(sigma_flat.device)

    dtau = (tau[1:] - tau[:-1]).view(1, 1, -1)
    p1   = torch.relu(-(c[:, :, 1:] - c[:, :, :-1]) / dtau).sum(dim=(1, 2))

    dm   = (m[1:] - m[:-1]).view(1, -1, 1)
    p2   = torch.relu((c[:, 1:, :] - c[:, :-1, :]) / dm).sum(dim=(1, 2))

    dm_l = (m[1:-1] - m[:-2]).view(1, -1, 1)
    dm_r = (m[2:]   - m[1:-1]).view(1, -1, 1)
    p3   = torch.relu(
        -(c[:, 2:, :] - c[:, 1:-1, :]) / dm_r
        + (c[:, 1:-1, :] - c[:, :-2, :]) / dm_l
    ).sum(dim=(1, 2))

    return p1 + p2 + p3


# ─────────────────────────────────────────────────────────────────────────────
# 7.  GRADIENT NORM MATCHING  — calibration de α_m et α_τ
# ─────────────────────────────────────────────────────────────────────────────

def _grad_norm(model: nn.Module) -> float:
    """Norme L2 de tous les gradients du modèle."""
    gs = [p.grad.data.flatten() for p in model.parameters() if p.grad is not None]
    return torch.cat(gs).norm().item() if gs else 0.0


def gradient_norm_matching(
    G:          Generator,
    D:          Discriminator,
    dataloader: DataLoader,
    norm:       Normalizer,
    n_epochs:   int = 25,
    device:     str = "cpu",
) -> tuple[float, float]:
    """
     Phase 1 du papier (Section 3.4) — version corrigée v3.

    Correction clé : un SEUL x_fake par batch pour les trois mesures
    de gradient (BCE, L_m, L_τ). Les ratios sont ainsi cohérents :
    les trois normes sont mesurées au même point du graphe de calcul,
    garantissant qu'elles correspondent au même tirage de G.

    Note : cette fonction modifie G et D. L'appelant DOIT avoir sauvegardé
    les state_dicts via copy.deepcopy avant l'appel, et les restaurer après.
    """
    bce_loss = nn.BCEWithLogitsLoss()
    opt_G = optim.RMSprop(G.parameters(), lr=1e-4)
    opt_D = optim.RMSprop(D.parameters(), lr=1e-4)
    ratios_m, ratios_tau = [], []

    for epoch in range(n_epochs):
        for cond_norm, x_norm in dataloader:
            cond_norm = cond_norm.to(device)
            x_norm    = x_norm.to(device)

            # ── Update D ──────────────────────────────────────────────────
            opt_D.zero_grad()
            with torch.no_grad():
                x_fake_d = G.sample(cond_norm)
            loss_D = (
                0.5 * bce_loss(D(cond_norm, x_norm),    torch.ones( len(x_norm),  1, device=device)) +
                0.5 * bce_loss(D(cond_norm, x_fake_d),  torch.zeros(len(x_fake_d),1, device=device))
            )
            loss_D.backward()
            opt_D.step()

            # ──  Un seul x_fake pour les trois mesures ─────────────────
            # On reconstruit g_{t+1} en espace original UNE SEULE FOIS
            x_fake  = G.sample(cond_norm)
            g_t     = norm.get_surface_from_condition(cond_norm)   # (B, 35)
            dg      = norm.inverse_dg(x_fake[:, 1:])               # (B, 35)
            g_next  = g_t.detach() + dg                            # gradient via dg

            # Norme gradient BCE
            opt_G.zero_grad()
            loss_bce = 0.5 * bce_loss(
                D(cond_norm, x_fake), torch.ones(len(x_fake), 1, device=device)
            )
            loss_bce.backward(retain_graph=True)
            grad_bce = _grad_norm(G)

            # Norme gradient L_m (même graphe)
            opt_G.zero_grad()
            Lm, Ltau = sobolev_penalties(g_next)
            Lm.backward(retain_graph=True)
            grad_Lm = _grad_norm(G)

            # Norme gradient L_τ (même graphe)
            opt_G.zero_grad()
            Ltau.backward()
            grad_Ltau = _grad_norm(G)

            if grad_Lm   > 1e-10: ratios_m.append(grad_bce / grad_Lm)
            if grad_Ltau > 1e-10: ratios_tau.append(grad_bce / grad_Ltau)

            # ── Update G (BCE only, comme dans le papier Section 3.4) ────
            opt_G.zero_grad()
            x_fake_g  = G.sample(cond_norm)
            loss_G_bce = 0.5 * bce_loss(
                D(cond_norm, x_fake_g), torch.ones(len(cond_norm), 1, device=device)
            )
            loss_G_bce.backward()
            opt_G.step()

    alpha_m   = float(np.clip(np.mean(ratios_m)   if ratios_m   else 1.0, 1e-3, 1e3))
    alpha_tau = float(np.clip(np.mean(ratios_tau) if ratios_tau else 1.0, 1e-3, 1e3))
    print(f"[GNM] α_m = {alpha_m:.4f}  |  α_τ = {alpha_tau:.4f}")
    return alpha_m, alpha_tau


# ─────────────────────────────────────────────────────────────────────────────
# 8.  BOUCLE D'ENTRAÎNEMENT PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

def _mode_collapse_score(
    G:                Generator,
    fixed_cond_norm:  torch.Tensor,
    n_samples:        int = 100,
    device:           str = "cpu",
) -> float:
    """
     Score de mode collapse = variance intra-condition des outputs.
    Calculé sur un batch fixe de conditions → stable d'une époque à l'autre.
    Score ~ 0 : G produit la même surface quel que soit le bruit z (pour une condition fixée) — mode collapse réel.
    Score ~ 1 : variance saine sur z en espace normalisé.
    """
    was_training = G.training
    G.eval()
    with torch.no_grad():
        B    = min(32, len(fixed_cond_norm))
        cond = fixed_cond_norm[:B].to(device)
        out  = G.sample(cond.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, DIM_A))
        score = out.view(B, n_samples, DIM_X).var(dim=1).mean().item()
    G.train(was_training)
    return score


def train_volgan(
    G:                    Generator,
    D:                    Discriminator,
    dataloader:           DataLoader,
    norm:                 Normalizer,
    alpha_m:              float,
    alpha_tau:            float,
    n_epochs:             int = 5000,
    device:               str = "cpu",
    collapse_check_every: int = 500,
    collapse_threshold:   float = 1e-4,
) -> dict:
    """
    Phase 2 du papier — Eq. (13)–(14).
    J_G = L_BCE + α_m·L_m + α_τ·L_τ
    Alternance 1:1 (1 update D / 1 update G par batch).
    """
    bce_loss = nn.BCEWithLogitsLoss()
    opt_G    = optim.RMSprop(G.parameters(), lr=1e-4)
    opt_D    = optim.RMSprop(D.parameters(), lr=1e-4)

    fixed_cond, _ = next(iter(dataloader))
    history = {"loss_G": [], "loss_D": [], "Lm": [], "Ltau": [], "collapse_score": []}

    for epoch in range(n_epochs):
        ep_lG, ep_lD, ep_lm, ep_lt = [], [], [], []

        for cond_norm, x_norm in dataloader:
            cond_norm = cond_norm.to(device)
            x_norm    = x_norm.to(device)
            B         = len(cond_norm)

            # ── (1) Update D ──────────────────────────────────────────────
            opt_D.zero_grad()
            with torch.no_grad():
                x_fake = G.sample(cond_norm)
            loss_D = (
                0.5 * bce_loss(D(cond_norm, x_norm),   torch.ones( B, 1, device=device)) +
                0.5 * bce_loss(D(cond_norm, x_fake),   torch.zeros(B, 1, device=device))
            )
            loss_D.backward()
            opt_D.step()

            # ── (2) Update G ──────────────────────────────────────────────
            opt_G.zero_grad()
            x_fake     = G.sample(cond_norm)
            loss_bce   = 0.5 * bce_loss(D(cond_norm, x_fake), torch.ones(B, 1, device=device))

            # Surface reconstruite en espace original pour les pénalités
            g_t     = norm.get_surface_from_condition(cond_norm)    # (B, 35)
            dg      = norm.inverse_dg(x_fake[:, 1:])                # (B, 35)
            g_next  = g_t.detach() + dg                             # gradient via dg

            Lm, Ltau = sobolev_penalties(g_next)
            loss_G   = loss_bce + alpha_m * Lm + alpha_tau * Ltau
            loss_G.backward()
            opt_G.step()

            ep_lG.append(loss_G.item()); ep_lD.append(loss_D.item())
            ep_lm.append(Lm.item());     ep_lt.append(Ltau.item())

        history["loss_G"].append(np.mean(ep_lG))
        history["loss_D"].append(np.mean(ep_lD))
        history["Lm"].append(np.mean(ep_lm))
        history["Ltau"].append(np.mean(ep_lt))

        if epoch % collapse_check_every == 0:
            sc     = _mode_collapse_score(G, fixed_cond, device=device)
            status = "⚠ MODE COLLAPSE" if sc < collapse_threshold else "OK"
            history["collapse_score"].append((epoch, sc))
            print(f"Epoch {epoch:5d} | "
                  f"L_G={history['loss_G'][-1]:.4f} | "
                  f"L_D={history['loss_D'][-1]:.4f} | "
                  f"L_m={history['Lm'][-1]:.5f} | "
                  f"L_τ={history['Ltau'][-1]:.5f} | "
                  f"collapse={sc:.4f} [{status}]")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# 9.  REPONDÉRATION DE SCÉNARIOS  (Weighted Monte Carlo)
# ─────────────────────────────────────────────────────────────────────────────

def scenario_reweighting(
    G:           Generator,
    norm:        Normalizer,
    cond_norm:   torch.Tensor,
    n_samples:   int   = 1000,
    beta_init:   float = 50.0,
    adaptive_beta: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Génère N scénarios et calcule w_i ∝ exp(−β·Λ(σ̂_i)).
    Eq. (17) du papier.

    adaptive_beta — approximation opérationnelle en deux étapes
    de la règle Eq. (20) : β(t) = 500 / max_i w_i(t).
    Étape 1 : calcul des poids bruts avec beta_init.
    Étape 2 : β_adapt = beta_init / max(w_bruts), recalcul des poids finaux.
    Ce n'est pas un point fixe exact, mais une heuristique stable.
    IMPORTANT : cette formule est équivalente à Eq. (20) uniquement si beta_init = 500
    (valeur par défaut recommandée). Changer beta_init modifie le régime de pénalisation
    sans suivre la calibration du papier.

    cond_norm : (DIM_A,)  — condition normalisée pour un seul jour
    returns   : (x_norm, x_orig, weights)  chacun de taille (N, DIM_X) / (N,)
    """
    was_training = G.training
    G.eval()
    with torch.no_grad():
        cond_exp = cond_norm.unsqueeze(0).expand(n_samples, -1)
        x_norm   = G.sample(cond_exp)                             # (N, DIM_X)
        x_orig   = norm.inverse_target(x_norm)

        g_t    = norm.get_surface_from_condition(cond_norm.unsqueeze(0)).squeeze(0)
        g_next = g_t + norm.inverse_dg(x_norm[:, 1:])
        sigma  = torch.exp(g_next).clamp(min=1e-4)
        pen    = arbitrage_penalty(sigma)                          # (N,)

        if adaptive_beta:
            lw0    = -(beta_init * pen - (beta_init * pen).max())
            w0     = torch.exp(lw0) / torch.exp(lw0).sum()
            beta   = float(np.clip(beta_init / (w0.max().item() + 1e-10), 1.0, 5000.0))

        beta = beta_init if not adaptive_beta else beta
        lw      = -(beta * pen - (beta * pen).max())
        weights = torch.exp(lw) / torch.exp(lw).sum()

    G.train(was_training)
    return x_norm, x_orig, weights


def weighted_expectation(v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (w.unsqueeze(-1) * v).sum(0) if v.dim() > 1 else (w * v).sum()


def weighted_quantile(v: torch.Tensor, w: torch.Tensor, q: float) -> torch.Tensor:
    """Quantile pondérée ; si aucun indice ne dépasse q (NaN/dégénéré), fallback min/max."""
    n = v.numel()
    if n == 0:
        return v
    idx = torch.argsort(v)
    w_sorted = w[idx].clone()
    w_sorted[torch.isnan(w_sorted)] = 0.0
    if w_sorted.sum() <= 0:
        w_sorted = torch.ones_like(w_sorted) / n
        warnings.warn("weighted_quantile: poids nuls ou NaN détectés, fallback uniforme.")
    cum = torch.cumsum(w_sorted, dim=0)
    k_candidates = (cum >= q).nonzero(as_tuple=True)[0]
    if k_candidates.numel() > 0:
        k = k_candidates[0].item()
        return v[idx[k]]
    return v[idx[-1]] if q >= 0.5 else v[idx[0]]


# ─────────────────────────────────────────────────────────────────────────────
# 10.  DONNÉES SYNTHÉTIQUES
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_surfaces(
    n_days: int   = 3000,
    kappa:  float = 2.0,
    theta:  float = 0.04,
    xi:     float = 0.3,
    rho:    float = -0.7,
    v0:     float = 0.04,
    dt:     float = 1/252,
    seed:   int   = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Surfaces stylisées issues d'une dynamique Heston discrète (Euler-Maruyama).
    σ(m,τ) ≈ σ_ATM(t) · [1 + skew(m,τ)] · term_struct(τ)
    Indexation : surfaces[t, _grid_index(i,j)]  ↔  (moneyness i, maturité j).
    """
    np.random.seed(seed)
    surfaces    = np.zeros((n_days, SURFACE_DIM))
    log_returns = np.zeros(n_days)
    v = v0
    for t in range(n_days):
        z1 = np.random.randn()
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.randn()
        R  = -0.5 * v * dt + np.sqrt(max(v, 0) * dt) * z1
        v  = max(v + kappa * (theta - v) * dt + xi * np.sqrt(max(v, 0) * dt) * z2, 1e-6)
        log_returns[t] = R
        s_atm = np.sqrt(v)
        for i, m in enumerate(MONEYNESS.numpy()):
            for j, tau in enumerate(MATURITIES.numpy()):
                s = max(s_atm * (1 - 0.15 * (m - 1) / np.sqrt(tau))
                               * (1 + 0.05 * np.log(max(tau, 1e-3))), 0.01)
                surfaces[t, _grid_index(i, j)] = np.log(s)
    return surfaces, log_returns


def build_dataset(
    surfaces:  np.ndarray,
    returns:   np.ndarray,
    lookback:  int = 21,
) -> tuple[TensorDataset, Normalizer, int]:
    """
    Construit le TensorDataset normalisé.
     Le Normalizer est fitté uniquement sur le train set.
    Retourne (dataset, normalizer, n_train).
    """
    conds_raw, tgts_raw = [], []
    n = len(returns)
    for t in range(lookback + 2, n):
        gamma = np.sqrt(252 / lookback * np.sum(returns[t - lookback:t]**2))
        conds_raw.append(np.concatenate([[returns[t-1], returns[t-2], gamma], surfaces[t-1]]))
        tgts_raw.append(np.concatenate([[returns[t]], surfaces[t] - surfaces[t-1]]))

    conds_raw = np.array(conds_raw, dtype=np.float32)
    tgts_raw  = np.array(tgts_raw,  dtype=np.float32)
    n_train   = int(0.8 * len(conds_raw))

    norm = Normalizer()
    norm.fit(conds_raw[:n_train], tgts_raw[:n_train])

    cond_t = torch.tensor(norm.transform_condition(conds_raw), dtype=torch.float32)
    tgt_t  = torch.tensor(norm.transform_target(tgts_raw),     dtype=torch.float32)
    return TensorDataset(cond_t, tgt_t), norm, n_train


# ─────────────────────────────────────────────────────────────────────────────
# 11.  ÉVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pca(
    G:               Generator,
    norm:            Normalizer,
    test_conditions: torch.Tensor,
    test_targets:    torch.Tensor,
    n_samples:       int = 10,
    device:          str = "cpu",
) -> dict:
    """
    Compare PCA des Δg réels vs simulés (espace original).
    Reproduit Table 4 + Figures 20-22 du papier.
    """
    from sklearn.decomposition import PCA

    # Δg réels en espace original
    dg_real = norm.inverse_dg(test_targets[:, 1:]).numpy()

    # Δg simulés (n_samples par condition)
    was_training = G.training
    G.eval()
    dg_sim = []
    with torch.no_grad():
        for cond in test_conditions:
            ce = cond.unsqueeze(0).expand(n_samples, -1).to(device)
            dg = norm.inverse_dg(G.sample(ce)[:, 1:]).cpu().numpy()
            dg_sim.append(dg)
    dg_sim = np.vstack(dg_sim)
    G.train(was_training)

    pca_r = PCA(n_components=3).fit(dg_real)
    pca_s = PCA(n_components=3).fit(dg_sim)
    ips   = [abs(float(pca_r.components_[k] @ pca_s.components_[k])) for k in range(3)]

    print("\n── PCA ──────────────────────────────────────────────────────────")
    print(f"{'Rang':<6} {'Données':>10} {'VolGAN':>10} {'Inner product':>15}")
    for k in range(3):
        print(f"PC{k+1:<4} {pca_r.explained_variance_ratio_[k]:>10.2%} "
              f"{pca_s.explained_variance_ratio_[k]:>10.2%} {ips[k]:>14.3f}")

    return {
        "var_data": pca_r.explained_variance_ratio_,
        "var_gan":  pca_s.explained_variance_ratio_,
        "inner_products": ips,
        "pca_real": pca_r,
        "components_data": pca_r.components_,
        "components_gan":  pca_s.components_,
    }


def evaluate_correlations(
    G:               Generator,
    norm:            Normalizer,
    test_conditions: torch.Tensor,
    test_targets:    torch.Tensor,
    pca_real:        object,
    n_mc:            int = 50,
    device:          str = "cpu",
) -> dict:
    """
     Corrélations de la LOI JOINTE simulée — pas des espérances conditionnelles.

    On empile TOUS les n_mc tirages de TOUTES les conditions pour former
    un nuage de points simulés de même taille que les données réelles × n_mc.
    Cela reproduit la structure jointe de (R_t, X¹_t, Δlog σ_ATM_t) telle
    qu'elle est générée par G, ce qui est l'objet statistique pertinent pour
    valider un modèle génératif — cf. Table 7 du papier.

    Comparer avec corr_data (corrélations des réalisations observées)
    est valide sur le fond (même loi jointe, sources différentes), mais les tailles
    d'échantillon diffèrent : n_test pour les données réelles, n_test × n_mc pour les
    simulations. La corrélation simulée est donc estimée avec une précision √n_mc fois
    supérieure — la comparaison reste informative mais n'est pas symétrique.
    """
    # ── Données réelles ───────────────────────────────────────────────────
    dg_real = norm.inverse_dg(test_targets[:, 1:]).numpy()
    R_real  = norm.inverse_Rt(test_targets[:, 0]).numpy()
    X1_real = dg_real @ pca_real.components_[0]
    ds_real = dg_real[:, ATM_IDX]
    corr_data = np.corrcoef(np.stack([R_real, X1_real, ds_real]))

    # ── Simulations — loi jointe  ─────────────────────────────────────
    was_training = G.training
    G.eval()
    R_all, X1_all, ds_all = [], [], []
    with torch.no_grad():
        for cond in test_conditions:
            ce   = cond.unsqueeze(0).expand(n_mc, -1).to(device)
            xf   = G.sample(ce)
            dg   = norm.inverse_dg(xf[:, 1:]).cpu().numpy()    # (n_mc, 35)
            Rmc  = norm.inverse_Rt(xf[:, 0]).cpu().numpy()
            # Empilement de tous les tirages (pas de moyenne) 
            R_all.extend(Rmc.tolist())
            X1_all.extend((dg @ pca_real.components_[0]).tolist())
            ds_all.extend(dg[:, ATM_IDX].tolist())
    G.train(was_training)

    corr_gan  = np.corrcoef(np.stack([R_all, X1_all, ds_all]))
    labels    = ["ΔlogS", "X¹(PC1)", "ΔlogσATM"]

    print("\n── Corrélations — loi jointe ────────────────────────────────────")
    print(f"  (n_data={len(R_real)}, n_sim={len(R_all)} = {len(test_conditions)}×{n_mc})")
    _print_corr("Données", corr_data, labels)
    _print_corr("VolGAN ",  corr_gan,  labels)
    return {"corr_data": corr_data, "corr_gan": corr_gan}


def _print_corr(name: str, C: np.ndarray, labels: list) -> None:
    print(f"\n  {name}    " + "".join(f"{l:>12}" for l in labels))
    for i, l in enumerate(labels):
        print(f"  {l:<12}" + "".join(f"{C[i,j]:>12.3f}" for j in range(len(labels))))


def evaluate_arbitrage_penalty(
    G:               Generator,
    norm:            Normalizer,
    test_conditions: torch.Tensor,
    test_targets:    torch.Tensor,
    n_mc:            int = 200,
    device:          str = "cpu",
) -> dict:
    """
     Pénalité d'arbitrage — version corrigée v3.

     Benchmark correct : on compare
        • données réelles   : g_{t+1}^real = g_t + Δg_t  (surface observée demain)
        • VolGAN généré     : g_{t+1}^fake = g_t + Δg_fake
    Ces deux objets sont homogènes — la v2 comparait g_t vs g_{t+1}^fake.

     Tenseurs de normalisation pré-convertis en dehors de la boucle.
    """
    # Pré-conversion sur CPU (compatible GPU si device change)
    dg_std_t  = torch.tensor(norm.dg_std,  dtype=torch.float32)
    dg_mean_t = torch.tensor(norm.dg_mean, dtype=torch.float32)
    g_std_t   = torch.tensor(norm.g_std,   dtype=torch.float32)
    g_mean_t  = torch.tensor(norm.g_mean,  dtype=torch.float32)

    # Surface réelle t+1 = g_t + Δg_t
    g_t_norm  = test_conditions[:, 3:]                            # normalisé
    g_t_orig  = (g_t_norm * g_std_t + g_mean_t).numpy()
    dg_t_norm = test_targets[:, 1:]
    dg_t_orig = dg_t_norm.numpy() * norm.dg_std + norm.dg_mean
    g_next_real  = g_t_orig + dg_t_orig                           # 
    sigma_real   = np.exp(g_next_real).astype(np.float32)
    pen_real     = arbitrage_penalty(torch.tensor(sigma_real)).numpy()

    # Surface GAN t+1 = g_t + Δg_fake
    was_training = G.training
    G.eval()
    pen_gan = []
    with torch.no_grad():
        g_t_t = torch.tensor(g_t_orig, dtype=torch.float32)       # (T, 35), CPU
        for b, cond in enumerate(test_conditions):
            ce      = cond.unsqueeze(0).expand(n_mc, -1).to(device)
            xf      = G.sample(ce)
            # Dénormalisation avec tenseurs pré-convertis
            dg_fake = xf[:, 1:].cpu() * dg_std_t + dg_mean_t     # (n_mc, 35)
            g_next  = g_t_t[b].unsqueeze(0) + dg_fake             # (n_mc, 35)
            sigma   = torch.exp(g_next).clamp(min=1e-4)
            pen_gan.append(arbitrage_penalty(sigma).mean().item())
    G.train(was_training)

    pen_gan = np.array(pen_gan)
    print("\n── Arbitrage Penalty  (g_{t+1} = g_t + Δg) ─────────────────────")
    print(f"{'':>26} {'Moyenne':>10} {'Écart-type':>12} {'Médiane':>10}")
    for label, arr in [("Données réelles (real)", pen_real),
                       ("VolGAN (avant reweight.)",  pen_gan)]:
        print(f"  {label:<24} {arr.mean():>10.4f} {arr.std():>12.4f} {np.median(arr):>10.4f}")
    return {"penalty_data": pen_real, "penalty_gan": pen_gan}


# ─────────────────────────────────────────────────────────────────────────────
# 12.  SCRIPT PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    set_seeds(42)   

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Grille : {N_M} × {N_TAU} = {SURFACE_DIM} pts  "
          f"(convention : g_flat[b, i*{N_TAU}+j])\n")

    # ── Tests unitaires ───────────────────────────────────────────────────
    _test_grid_indexing()

    # ── Données synthétiques ──────────────────────────────────────────────
    print("\nGénération des données (Heston stylisé)...")
    surfaces, returns = generate_synthetic_surfaces(n_days=3000, seed=42)

    # ── Dataset + normalisation train-only  ─────────────────────────
    dataset, norm, n_train = build_dataset(surfaces, returns)
    n_total = len(dataset)
    n_test  = n_total - n_train

    train_ds = Subset(dataset, list(range(n_train)))
    test_ds  = Subset(dataset, list(range(n_train, n_total)))
    print(f"Train : {n_train} obs  |  Test : {n_test} obs  (split chronologique)")

    train_loader = DataLoader(train_ds, batch_size=100, shuffle=True)  # 
    test_conditions = torch.stack([test_ds[i][0] for i in range(n_test)])
    test_targets    = torch.stack([test_ds[i][1] for i in range(n_test)])

    # ── Modèles ───────────────────────────────────────────────────────────
    G = Generator(latent_dim=32, hidden=16).to(device)
    D = Discriminator(hidden=16).to(device)
    print(f"\nGénérateur     : {sum(p.numel() for p in G.parameters()):,} params")
    print(f"Discriminateur : {sum(p.numel() for p in D.parameters()):,} params")

    # ── Phase 1 : GNM ─────────────────────────────────────────────────────
    #  Sauvegarde AVANT la Phase 1
    G_init = copy.deepcopy(G.state_dict())
    D_init = copy.deepcopy(D.state_dict())

    print("\n=== Phase 1 : Gradient Norm Matching ===")
    alpha_m, alpha_tau = gradient_norm_matching(
        G, D, train_loader, norm, n_epochs=25, device=device
    )

    #  Restauration pour Phase 2 — MÊME initialisation
    G.load_state_dict(G_init)
    D.load_state_dict(D_init)
    print("[GNM] Initialisation restaurée.\n")

    # ── Phase 2 : Entraînement complet ───────────────────────────────────
    print("=== Phase 2 : Entraînement VolGAN ===")
    history = train_volgan(
        G, D, train_loader, norm,
        alpha_m=alpha_m, alpha_tau=alpha_tau,
        n_epochs=5000, device=device,
        collapse_check_every=500,
        collapse_threshold=1e-4,
    )

    # ── Évaluation out-of-sample ──────────────────────────────────────────
    print("\n=== Évaluation out-of-sample ===")

    pca_res  = evaluate_pca(G, norm, test_conditions, test_targets,
                            n_samples=200, device=device)  # n_samples=200 pour main() ; le notebook utilise 1000
    corr_res = evaluate_correlations(G, norm, test_conditions, test_targets,
                                     pca_real=pca_res["pca_real"],
                                     n_mc=50, device=device)
    arb_res  = evaluate_arbitrage_penalty(G, norm, test_conditions, test_targets,
                                          n_mc=200, device=device)

    # ── Exemple de repondération ──────────────────────────────────────────
    print("\n=== Repondération — premier jour du test set ===")
    xn, xo, w = scenario_reweighting(
        G, norm, test_conditions[0].to(device),
        n_samples=1000, beta_init=500.0, adaptive_beta=True
    )
    R_vec = xo[:, 0]
    print(f"  E_β[R_t]   = {weighted_expectation(R_vec, w).item():.6f}")
    print(f"  IC 95%     = [{weighted_quantile(R_vec, w, 0.025).item():.4f},"
          f" {weighted_quantile(R_vec, w, 0.975).item():.4f}]")
    print(f"  ESS        = {1.0/(w**2).sum().item():.1f}  (sur 1000 scénarios)")

    return G, D, norm, history, pca_res, corr_res, arb_res


if __name__ == "__main__":
    main()