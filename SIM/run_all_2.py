import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from bcqm3_sim import (
    Params, simulate_chain,
    ensemble_mean_displacement, autocorr_sweep
)

BASE = Path(__file__).resolve().parent
FIGS = BASE / "figs"
OUTS = BASE / "outputs"
FIGS.mkdir(exist_ok=True, parents=True)
OUTS.mkdir(exist_ok=True, parents=True)

# ---- Shared style helper ----
FIGSIZE = (9, 6)
def _style_axes(ax, title, xlabel, ylabel):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.minorticks_on()
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.8)
    ax.grid(False, which="minor", axis="both") # (optional hard-off for minors)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

# ---------- Fig. 1 ----------
def fig1_trajectories():
    p = Params(seed=20251106, tau0=1.0, beta=2.0, kappa=1e-3)
    W = 100.0
    RUNS = 80
    STEPS = 220

    xs = []
    for r in range(RUNS):
        p2 = Params(**{**p.__dict__, "seed": p.seed + 1000*r})
        x, _ = simulate_chain(STEPS, Wcoh=W, b=0.0, p=p2, init_dir=+1)
        xs.append(x)

    arr = np.stack(xs, axis=0)
    t = np.arange(STEPS+1) * p.dt
    mean_x = arr.mean(axis=0)
    low = np.percentile(arr, 2.5, axis=0)
    high = np.percentile(arr, 97.5, axis=0)

    plt.figure(figsize=FIGSIZE)
    ax = plt.gca()
    for j in range(min(RUNS, 40)):
        ax.plot(t, arr[j, :], linewidth=0.8)
    ax.plot(t, mean_x, linewidth=2.0, label="mean")
    ax.fill_between(t, low, high, alpha=0.2, label="95% band")
    _style_axes(ax, r"Fig. 1  Trajectory ensembles at $b=0$", r"$t$", r"$x(t)$")
    ax.legend()
    outp = FIGS / "fig1_trajectories.pdf"
    plt.tight_layout(); plt.savefig(outp); plt.close()
    print("Wrote", outp)

    # CSV for reproducibility
    with open(OUTS / "fig1_trajectories.csv", "w", newline="") as fh:
        cols = ["t"] + [f"x_run_{i}" for i in range(RUNS)] + ["mean_x","ci_low","ci_high"]
        w = csv.writer(fh); w.writerow(cols)
        for i in range(len(t)):
            row = [t[i]] + [arr[j, i] for j in range(RUNS)] + [mean_x[i], low[i], high[i]]
            w.writerow(row)

# ---------- Fig. 2 ----------
def fig2_mean_displacement():
    p = Params(seed=20251106, tau0=1.0, beta=2.0, kappa=1e-3)
    T_STEPS = 200
    RUNS = 256
    W = 100.0
    t, mean_x = ensemble_mean_displacement(T_STEPS, RUNS, Wcoh=W, p=p)

    plt.figure(figsize=FIGSIZE)
    ax = plt.gca()
    ax.plot(t, mean_x, "-", linewidth=2.0)
    _style_axes(ax, r"Fig. 2  Mean displacement $\langle \Delta x(t)\rangle$ at $b=0$", r"$t$", r"$\langle \Delta x(t)\rangle$")
    outp = FIGS / "fig2_mean_displacement.pdf"
    plt.tight_layout(); plt.savefig(outp); plt.close()
    print("Wrote", outp)

    with open(OUTS / "fig2_mean_displacement.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["t", "mean_x"])
        for i in range(len(t)):
            w.writerow([t[i], mean_x[i]])

# ---------- Fig. 3 (robust slope, styled) ----------
def fig3_meff_vs_Wcoh():
    # Hold persistence ~ independent of W for this panel (clean ~ -2) but keep robust slope & masking
    p = Params(seed=20251106, tau0=1e-9, beta=2.0, kappa=2e-5)
    W_list = [30, 50, 80, 120, 180, 260]
    N, burn, runs = 8000, 800, 200
    b0 = 3e-3

    def mean_v(W, b, rr=runs):
        vals = []
        for r in range(rr):
            p2 = Params(**{**p.__dict__, "seed": p.seed + 10000*r})
            _, v = simulate_chain(N, Wcoh=W, b=b, p=p2, init_dir=+1)
            vals.append(np.mean(v[burn:]))
        return float(np.mean(vals))

    xs, ys = [], []
    for W in W_list:
        bs = np.array([-b0, -b0/2, +b0/2, +b0], float)
        mus = np.array([mean_v(W, b) for b in bs], float)
        A = np.vstack([bs, np.ones_like(bs)]).T
        slope, _ = np.linalg.lstsq(A, mus, rcond=None)[0]
        if slope <= 0.0:
            mus = np.array([mean_v(W, b, rr=runs*2) for b in bs], float)
            slope, _ = np.linalg.lstsq(A, mus, rcond=None)[0]
        meff = (1.0 / slope) if slope > 0.0 else np.nan
        xs.append(W); ys.append(meff)

    xs = np.array(xs, float)
    ys = np.array(ys, float)
    mask = np.isfinite(ys) & (ys > 0)
    if not np.all(mask):
        dropped = xs[~mask].tolist()
        print("Fig.3 dropped Wcoh (nonpositive/NaN m_eff):", dropped)

    # Fit & plot
    lx, ly = np.log(xs[mask]), np.log(ys[mask])
    A = np.vstack([lx, np.ones_like(lx)]).T
    m, c = np.linalg.lstsq(A, ly, rcond=None)[0]

    plt.figure(figsize=FIGSIZE)
    ax = plt.gca()
    ax.loglog(xs[mask], ys[mask], "o-",
              label="est.")
    xf = np.linspace(xs[mask].min(), xs[mask].max(), 200)
    ax.loglog(xf, np.exp(c)*xf**m, "--",
              label=f"fit slope={m:.2f}")
    _style_axes(ax, r"Fig. 3  $m_{\mathrm{eff}}$ vs $W_{\mathrm{coh}}$", r"$W_{\mathrm{coh}}$", r"$m_{\mathrm{eff}}$ (est.)")
    ax.legend()
    outp = FIGS / "fig3_meff_vs_Wcoh.pdf"
    plt.tight_layout(); plt.savefig(outp); plt.close()
    print("Wrote", outp)

    # Save table for appendix
    with open(OUTS / "fig3_meff_values.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["Wcoh","m_eff_est"])
        for W, M in zip(xs[mask], ys[mask]):
            w.writerow([float(W), float(M)])

# ---------- Fig. 4 ----------
def fig4_autocorr_collapse():
    p = Params(seed=20251106, tau0=1.0, beta=2.0, kappa=1e-3)
    W_list = [30, 60, 120, 240]
    results = autocorr_sweep(Wcoh_list=W_list, p=p, N=6000, burn=400, max_lag=1200)

    plt.figure(figsize=FIGSIZE)
    ax = plt.gca()
    for W, rec in results.items():
        scaled = rec["lag"] / float(W)
        ax.plot(scaled, rec["ac"], label=f"Wcoh={W}")
    _style_axes(ax, r"Fig. 4  Velocity autocorrelation collapse", r"$\mathrm{lag}/W_{\mathrm{coh}}$", r"$C_v(\mathrm{lag})$")
    ax.legend()
    outp = FIGS / "fig4_autocorr_collapse.pdf"
    plt.tight_layout(); plt.savefig(outp); plt.close()
    print("Wrote", outp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["fig1","fig2","fig3","fig4"], help="Build only one figure")
    args = ap.parse_args()
    if args.only:
        {"fig1": fig1_trajectories,
         "fig2": fig2_mean_displacement,
         "fig3": fig3_meff_vs_Wcoh,
         "fig4": fig4_autocorr_collapse}[args.only]()
    else:
        fig1_trajectories()
        fig2_mean_displacement()
        fig3_meff_vs_Wcoh()
        fig4_autocorr_collapse()

if __name__ == "__main__":
    main()
