from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import math

@dataclass
class Params:
    a: float = 1.0
    dt: float = 1.0
    tau0: float = 1.0     # persistence factor: tau = tau0 * Wcoh
    beta: float = 2.0     # window factor controlling S(Wcoh)
    kappa: float = 1e-3   # scale for Δp = kappa * b * S(Wcoh)
    seed: int = 20251106

def _S_from_Wcoh(Wcoh: float, beta: float) -> float:
    """Geometric-series factor ~ Wcoh^2 (used for small-drive scaling)."""
    c = 0.5 + 1.0 / beta
    q = math.exp(-c / max(Wcoh, 1e-9))
    d = (1.0 - q)
    if d <= 1e-12:
        return 1.0 / (1e-12**2)
    return q / (d*d)

def _delta_p(Wcoh: float, b: float, p: Params) -> float:
    S = _S_from_Wcoh(Wcoh, p.beta)
    dp = p.kappa * b * S
    return max(-0.25, min(0.25, dp))  # clamp so probabilities remain sane

def _persist_prob(Wcoh: float, p: Params) -> float:
    tau = max(1e-6, p.tau0 * Wcoh)
    return math.exp(-p.dt / tau)

def simulate_chain(N: int, Wcoh: float, b: float, p: Params, init_dir: int = +1):
    """
    Minimal 1D event-chain: s_n in {-1,+1}, x accumulates ±a each step.
    With prob 'persist' keep direction; otherwise resample with drift Δp.
    Returns (x positions of length N+1, step velocities of length N).
    """
    rng = np.random.default_rng(p.seed + int(Wcoh*1000) + int(b*1e6) + N)
    x = np.zeros(N+1, dtype=float)
    v = np.zeros(N, dtype=float)
    s = +1 if init_dir >= 0 else -1
    dp = _delta_p(Wcoh, b, p)
    pp = _persist_prob(Wcoh, p)
    stepv = p.a / p.dt
    for n in range(N):
        if rng.random() > pp:
            s = +1 if rng.random() < 0.5*(1.0 + dp) else -1
        v[n] = stepv * s
        x[n+1] = x[n] + p.a * s
    return x, v

# --- Helpers used by figure builders ---

def ensemble_mean_displacement(N: int, runs: int, Wcoh: float, p: Params):
    xs = []
    for r in range(runs):
        p2 = Params(**{**p.__dict__, "seed": p.seed + 1000*r})
        x, _ = simulate_chain(N, Wcoh=Wcoh, b=0.0, p=p2, init_dir=+1)
        xs.append(x)
    arr = np.stack(xs, axis=0)
    t = np.arange(N+1) * p.dt
    return t, arr.mean(axis=0)

def _vel_autocorr(v: np.ndarray, max_lag: int) -> np.ndarray:
    v = v - v.mean()
    denom = float(v @ v) if float(v @ v) != 0 else 1.0
    ac = np.zeros(max_lag+1, dtype=float)
    ac[0] = 1.0
    for lag in range(1, max_lag+1):
        ac[lag] = (v[:-lag] @ v[lag:]) / denom
    return ac

def autocorr_sweep(Wcoh_list, p: Params, N=4000, burn=200, max_lag=600):
    results = {}
    for W in Wcoh_list:
        p2 = Params(**{**p.__dict__, "seed": p.seed + int(W*100)})
        _, v = simulate_chain(N, Wcoh=W, b=0.0, p=p2, init_dir=+1)
        v_use = v[burn:]
        ac = _vel_autocorr(v_use, max_lag=min(max_lag, len(v_use)//2))
        results[W] = {"lag": np.arange(ac.shape[0]), "ac": ac}
    return results
