# BCQM III — Simulation Code

This repository contains the simulation code used for the numerical results in

> Boundary-Condition Quantum Mechanics III (BCQM III):  
> stochastic event chains and emergent inertia

The code implements a minimal 1D event-chain model in which:

- A particle takes steps \(s_n \in \{-1,+1\}\) on a lattice of spacing \(a\).
- The **coherence horizon** \(W_{\mathrm{coh}}\) sets a persistence time
  \(\tau = \tau_0 W_{\mathrm{coh}}\) for the step direction.
- A small **drive** \(b\) (dimensionless) enters through a drift in the
  left/right choice probabilities,
  \[
    \Delta p = \kappa\, b\, S(W_{\mathrm{coh}}),
  \]
  where \(S(W_{\mathrm{coh}})\) is a geometric-series factor scaling
  as \(S(W_{\mathrm{coh}}) \propto W_{\mathrm{coh}}^2\) for large \(W_{\mathrm{coh}}\).
- The resulting trajectories are used to estimate:
  - ensemble trajectory clouds (Fig. 1),
  - mean displacement at zero drive (Fig. 2),
  - an effective inertial mass \(m_{\mathrm{eff}}(W_{\mathrm{coh}})\) from
    response to small \(b\) (Fig. 3),
  - velocity autocorrelation functions and their collapse when the lag
    is rescaled by \(W_{\mathrm{coh}}\) (Fig. 4).

All figures and CSV outputs in the BCQM III paper are reproducible from this code
(up to Monte Carlo noise, with fixed seeds for reproducibility).

---

## Repository layout

Assuming the code lives in a folder called `SIM/` inside the repo, you should see:

```text
BCQM_III/
  SIM/
    bcqm3_sim.py        # core simulation model
    run_all.py          # main driver to regenerate all figures and CSVs
    run_all_2.py        # optional variant with LaTeX-style labels in plots
    figs/               # PDF figures (generated or pre-populated)
      fig1_trajectories.pdf
      fig2_mean_displacement.pdf
      fig3_meff_vs_Wcoh.pdf
      fig4_autocorr_collapse.pdf
    outputs/            # CSVs for reproducibility
      fig1_trajectories.csv
      fig2_mean_displacement.csv
      fig3_meff_values.csv
      
      
 ---

## TL;DR

```bash
git clone https://github.com/PMF57/BCQM_III.git
cd BCQM_III/SIM
pip install numpy matplotlib      # or: pip install -r requirements.txt
python run_all.py                 # regenerates all figures and CSVs