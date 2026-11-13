# Asymmetric regularization for GAN training: numerical experiments

Code for the numerical section of

> S. C. Giagtzoglou, M. H. M. Winands and B. Franci, "Asymmetric Regularization for GAN Training via Operator Theory", submitted to the Learning for Dynamics and Control Conference (L4DC), 2027.

## Overview

The paper studies GAN training as a variational inequality (VI) for the saddle operator of a zero-sum game, with a penalty applied on the discriminator side only. Two penalties are considered: an asymmetric Tikhonov term, $B=\tfrac12\|\varphi\|^2$, and a zero-centered input-gradient penalty (ZC), $B=\tfrac12\mathbb E_x\|\nabla_x D_\varphi(x)\|^2+\tfrac12\mathbb E_z\|\nabla_x D_\varphi(G_\theta(z))\|^2$. The regularized objective is $\mathcal L-\gamma B$ for both players, so the game stays zero-sum.

This repository contains the two examples used to illustrate the analysis. Both are small enough that all expectations are computed by quadrature, so the experiments are deterministic apart from the choice of initial points.

## Repository structure

| File | Content |
|---|---|
| `main.py` | Model, algorithms, experiments and figures |
| `requirements.txt` | Python dependencies |
| `results_paper/` | Output of the run reported in the paper |
| `results_paper/multistart_paths.csv` | Residual of $F_\gamma$, residual of $F$ and Nash gap along all 750 runs |
| `results_paper/multistart_summary.csv` | Counts and oracle calls per configuration and algorithm |
| `results_paper/table_stepsizes.{csv,tex}` | Lipschitz estimates and step sizes (Table 1) |
| `results_paper/fig_bilinear.pdf`, `fig_multistart.pdf` | Figures 1 and 2 |

## Installation

Python 3.9 or later with NumPy and Matplotlib:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py selftest             # gradient, quadrature and Nash-gap checks (seconds)
python main.py bilinear             # Figure 1
python main.py stepsizes            # Table 1
python main.py multistart --jobs 6  # 750 runs and Figure 2
python main.py plot --csv results_paper/multistart_paths.csv   # redraw Figure 2 from the stored runs
```

`multistart --quick` runs 4 initial points for 1500 iterations and is meant only as a test of the pipeline. `--configs tik1,zc1` restricts the figure to some configurations, and `--residual-ymin`, `--gap-ymin` set the lower axis limits (both $10^{-4}$ by default, as in the paper). `python main.py --help` lists the remaining options.

The multi-start study took about three minutes with six processes on a laptop; with one process it takes correspondingly longer.

## Inputs and outputs

The script needs no input data. Outputs are written to `results/` (`--out` changes the folder):

| File | Content |
|---|---|
| `fig_bilinear.pdf` | Trajectories of the bilinear example |
| `table_stepsizes.csv`, `.tex` | $\hat L$ and the step sizes |
| `multistart_paths.csv` | One row per run and quantity (`residual_Fgamma`, `residual_F`, `nash_gap`); the columns `it_K` hold the value after `K` iterations, at 102 recorded iterations. After a run stops, its final value is repeated |
| `multistart_summary.csv` | Per configuration and algorithm: runs below the stopping tolerance, runs with Nash gap at most $10^{-4}$, medians at the end, and oracle calls needed to reach that gap |
| `fig_multistart.pdf` | Median and 10%-90% quantiles of the residual of $F_\gamma$ (top) and of the Nash gap (bottom) against oracle calls |

Configurations are labelled `none`, `tik1`, `tik3`, `zc1`, `zc3`.

## Reproducibility

All random choices are seeded: the initial points of the nonlinear example (seed 0), the pairs of the Lipschitz estimate (seed 42) and the initial point of the bilinear example (seed 7). The run in `results_paper/` was produced with Python 3.12 on Windows. Rerunning 45 of the 750 runs with this script on Linux (Python 3.13, NumPy 2.5, Matplotlib 3.11) reproduced the stored values to the three significant digits kept in the CSV file; the other runs were not repeated. Results may differ in the last digits across platforms and library versions, since runs that stop near the tolerance depend on floating-point rounding.

In that run, all 300 Tikhonov runs reach the stopping tolerance and none ends with a Nash gap below $10^{-4}$; 292 of the 300 ZC runs end with a gap below $10^{-4}$. The counts for each configuration and algorithm are in `results_paper/multistart_summary.csv`.
