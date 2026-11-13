#!/usr/bin/env python3
"""
Usage:
  python main.py selftest
  python main.py bilinear
  python main.py stepsizes
  python main.py multistart [--jobs N] [--quick]
  python main.py plot [--csv PATH]
  python main.py all

Outputs are written to ./results (change with --out).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8,
                     "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})

# ----------------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------------
SIGMA_X, SIGMA_Z = 1.0, 0.5
BOX = 3.0
GAP_GRID = np.linspace(-BOX, BOX, 601)
ALGORITHMS = ("FB", "EG", "EFTP")
CONFIGS = (("none", 0.0), ("tik", 1.0), ("tik", 3.0), ("zc", 1.0), ("zc", 3.0))
STOP_TOL = 1e-10
NASH_TOL = 1e-4
PLOT_FLOOR = 1e-12
QUANTITIES = ("residual_Fgamma", "residual_F", "nash_gap")
PLOT_STYLE = {"none": ("No reg", "0.45", ":"),
              "tik1": (r"Tik $\gamma=1$", "C0", "--"),
              "tik3": (r"Tik $\gamma=3$", "C1", "--"),
              "zc1": (r"ZC $\gamma=1$", "C2", "-"),
              "zc3": (r"ZC $\gamma=3$", "C3", "-")}


def config_tag(config):
    """Short name of a configuration: none, tik1, tik3, zc1, zc3."""
    penalty, gamma = config
    return penalty if penalty == "none" else f"{penalty}{gamma:g}"


# ----------------------------------------------------------------------------------------------
# Nonlinear example: quadrature, loss, penalties, saddle operator, Nash gap
# ----------------------------------------------------------------------------------------------
def gauss_trapezoid(n_nodes=401, width=10.0):
    """Nodes and normalized weights of the trapezoid rule for E[f(u)], u ~ N(0, 1)."""
    u = np.linspace(-width, width, n_nodes)
    w = np.exp(-0.5 * u ** 2)
    return u, w / w.sum()


class NonlinearGame:
    """D = phi1 tanh(phi2 x), G = theta + z, omega = (theta, phi1, phi2), penalty none/tik/zc."""

    def __init__(self, penalty="none", gamma=0.0, n_nodes=401):
        if penalty not in ("none", "tik", "zc"):
            raise ValueError(f"unknown penalty {penalty!r}")
        self.penalty, self.gamma = penalty, float(gamma)
        u, self.weights = gauss_trapezoid(n_nodes)
        self.x_nodes, self.z_nodes = SIGMA_X * u, SIGMA_Z * u

    @staticmethod
    def project(omega):
        return np.clip(omega, -BOX, BOX)

    def expect(self, values):
        return float(self.weights @ values)

    def loss(self, omega):
        theta, phi1, phi2 = omega
        return (phi1 * self.expect(np.tanh(phi2 * self.x_nodes))
                - phi1 * self.expect(np.tanh(phi2 * (theta + self.z_nodes))))

    def grad_loss(self, omega):
        theta, phi1, phi2 = omega
        fake = theta + self.z_nodes
        sech2_real = 1.0 / np.cosh(phi2 * self.x_nodes) ** 2
        sech2_fake = 1.0 / np.cosh(phi2 * fake) ** 2
        return np.array([
            -phi1 * phi2 * self.expect(sech2_fake),
            self.expect(np.tanh(phi2 * self.x_nodes)) - self.expect(np.tanh(phi2 * fake)),
            phi1 * (self.expect(self.x_nodes * sech2_real) - self.expect(fake * sech2_fake))])

    def penalty_value(self, omega):
        theta, phi1, phi2 = omega
        if self.penalty == "none":
            return 0.0
        if self.penalty == "tik":
            return 0.5 * (phi1 ** 2 + phi2 ** 2)
        sech4_real = 1.0 / np.cosh(phi2 * self.x_nodes) ** 4
        sech4_fake = 1.0 / np.cosh(phi2 * (theta + self.z_nodes)) ** 4
        return 0.5 * phi1 ** 2 * phi2 ** 2 * (self.expect(sech4_real) + self.expect(sech4_fake))

    def grad_penalty(self, omega):
        theta, phi1, phi2 = omega
        if self.penalty == "none":
            return np.zeros(3)
        if self.penalty == "tik":
            return np.array([0.0, phi1, phi2])

        def moments(t):
            sech4, tanh = 1.0 / np.cosh(phi2 * t) ** 4, np.tanh(phi2 * t)
            return self.expect(sech4), self.expect(tanh * sech4), self.expect(t * tanh * sech4)

        s4_real, _, tts4_real = moments(self.x_nodes)
        s4_fake, ts4_fake, tts4_fake = moments(theta + self.z_nodes)
        return np.array([
            -2.0 * phi1 ** 2 * phi2 ** 3 * ts4_fake,
            phi1 * phi2 ** 2 * (s4_real + s4_fake),
            phi1 ** 2 * phi2 * (s4_real + s4_fake) - 2.0 * phi1 ** 2 * phi2 ** 2 * (tts4_real + tts4_fake)])

    def operator(self, omega):
        """F_gamma = (dL/dtheta - gamma dB/dtheta, -dL/dphi + gamma dB/dphi)."""
        gl, gb, gamma = self.grad_loss(omega), self.grad_penalty(omega), self.gamma
        return np.array([gl[0] - gamma * gb[0], -gl[1] + gamma * gb[1], -gl[2] + gamma * gb[2]])

    def nash_gap(self, omega):
        """max_phi' L(theta, phi') - min_theta' L(theta', phi), maxima on GAP_GRID."""
        theta, phi1, phi2 = omega
        m_over_phi2 = np.tanh(np.outer(GAP_GRID, theta + self.z_nodes)) @ self.weights
        m_over_theta = np.tanh(phi2 * (GAP_GRID[:, None] + self.z_nodes[None, :])) @ self.weights
        return float(BOX * np.max(np.abs(m_over_phi2)) + abs(phi1) * np.max(np.abs(m_over_theta)))


def natural_residual(operator, omega):
    """|| omega - Pi_S(omega - F(omega)) ||."""
    return float(np.linalg.norm(omega - np.clip(omega - operator(omega), -BOX, BOX)))


def lipschitz_estimate(game, n_pairs=4000, seed=42):
    """Largest ratio ||F(u) - F(v)|| / ||u - v|| over random pairs in S."""
    rng, best = np.random.default_rng(seed), 0.0
    for _ in range(n_pairs):
        u, v = rng.uniform(-BOX, BOX, 3), rng.uniform(-BOX, BOX, 3)
        dist = np.linalg.norm(u - v)
        if dist > 1e-12:
            best = max(best, np.linalg.norm(game.operator(u) - game.operator(v)) / dist)
    return best


def step_sizes(n_pairs=4000):
    """Step sizes from the Lipschitz estimate L for each configuration."""
    table = {}
    for config in CONFIGS:
        lip = lipschitz_estimate(NonlinearGame(*config), n_pairs)
        table[config] = {"Lhat": lip, "FB": min(0.9 / (1.05 * lip), 0.5), "EG": 1.0 / (1.05 * lip),
                         "EFTP": 1.0 / (4.0 * 1.05 * lip)}
    return table


# ----------------------------------------------------------------------------------------------
# Algorithms
# ----------------------------------------------------------------------------------------------
def checkpoint_grid(n_iter, n_log=70, n_lin=41):
    """Iterations at which the diagnostics are recorded."""
    k = np.concatenate([[0.0], np.round(np.logspace(0, np.log10(max(n_iter, 1)), n_log)),
                        np.round(np.linspace(0, n_iter, n_lin))])
    return np.unique(k.astype(int))


def oracle_calls(algorithm, iterations):
    """Evaluations of F_gamma after the given number of iterations."""
    iterations = np.asarray(iterations)
    return {"FB": iterations, "EG": 2 * iterations, "EFTP": iterations + 1}[algorithm]


def run_algorithm(game, reference, start, algorithm, eta, n_iter, checkpoints,
                  stop_tol=STOP_TOL, check_every=10):
    """Projected FB, EG or EFTP on VI(S, F_gamma).

    Records the residual of F_gamma, the residual of F and the Nash gap of `reference` at the
    checkpoints; after a stop the final values are held. Returns (last iterate, iterations, record).
    """
    omega = game.project(np.asarray(start, float))
    stored = game.operator(omega.copy()) if algorithm == "EFTP" else None
    record = np.full((3, len(checkpoints)), np.nan)

    def diagnostics(point):
        return (natural_residual(game.operator, point), natural_residual(reference.operator, point),
                reference.nash_gap(point))

    j = 0
    if checkpoints[0] == 0:
        record[:, 0] = diagnostics(omega)
        j = 1
    k = 0
    for k in range(1, n_iter + 1):
        if algorithm == "FB":
            omega = game.project(omega - eta * game.operator(omega))
        elif algorithm == "EG":
            extrapolated = game.project(omega - eta * game.operator(omega))
            omega = game.project(omega - eta * game.operator(extrapolated))
        else:
            omega = game.project(omega - eta * stored)
            extrapolated = game.project(omega - eta * stored)
            stored = game.operator(extrapolated)
        if j < len(checkpoints) and k == checkpoints[j]:
            record[:, j] = diagnostics(omega)
            j += 1
        if (k % check_every == 0 or k == n_iter) and natural_residual(game.operator, omega) < stop_tol:
            break
    if j < len(checkpoints):
        record[:, j:] = np.array(diagnostics(omega))[:, None]
    return omega, k, record


# ----------------------------------------------------------------------------------------------
# Bilinear example
# ----------------------------------------------------------------------------------------------
def bilinear_trajectory(algorithm, curvature, start, a=1.0, box=4.0, n_iter=120):
    """Iterates of projected FB, EG or EFTP on F(w) = J w, J = [[0, a], [-a, curvature]]."""
    jac = np.array([[0.0, a], [-a, curvature]])
    lip = float(np.linalg.svd(jac, compute_uv=False)[0])
    eta = {"FB": 0.95 / max(lip, 1.0), "EG": 1.0 / (1.05 * lip), "EFTP": 1.0 / (4.0 * lip)}[algorithm]

    def project(w):
        return np.clip(w, -box, box)

    w = start.copy()
    path = [w.copy()]
    stored = jac @ w
    for _ in range(n_iter):
        if algorithm == "FB":
            w = project(w - eta * (jac @ w))
        elif algorithm == "EG":
            w = project(w - eta * (jac @ project(w - eta * (jac @ w))))
        else:
            w = project(w - eta * stored)
            stored = jac @ project(w - eta * stored)
        path.append(w.copy())
    return np.array(path), eta


def figure_bilinear(out_dir, width=6.0, height=1.75, gamma=2.0, box=4.0):
    """Trajectories of FB, EG and EFTP with and without Tikhonov regularization."""
    start = np.random.default_rng(7).normal(size=2)
    grey = "0.55"
    fig, axes = plt.subplots(1, 4, figsize=(width, height), gridspec_kw={"width_ratios": [1.15, 1, 1, 1]})

    def draw(ax, path, colour, lw=0.8, zorder=2):
        ax.plot(path[:, 0], path[:, 1], color=colour, lw=lw, zorder=zorder)
        ax.plot(*path[0], "o", ms=3.5, color=colour, zorder=zorder + 1)
        ax.plot(*path[-1], "s", ms=3.5, color=colour if colour == grey else "C3", zorder=zorder + 1)

    path, eta = bilinear_trajectory("FB", 0.0, start, box=box)
    ax = axes[0]
    ax.plot([-box, box, box, -box, -box], [-box, -box, box, box, -box], ls="--", lw=0.6, color="0.5", zorder=1)
    draw(ax, path, "C0")
    ax.set_xlim(-1.1 * box, 1.1 * box)
    ax.set_ylim(-1.1 * box, 1.1 * box)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("(a) FB, no reg.")
    print(f"(a) FB   no reg : eta = {eta:.4f}, |w_T| = {np.linalg.norm(path[-1]):.2e}")

    for ax, label, algorithm in zip(axes[1:], ("(b)", "(c)", "(d)"), ALGORITHMS):
        path, eta = bilinear_trajectory(algorithm, gamma, start, box=box)
        if algorithm != "FB":
            path0, eta0 = bilinear_trajectory(algorithm, 0.0, start, box=box)
            draw(ax, path0, grey, lw=0.7, zorder=1)
            print(f"{label} {algorithm:4s} no reg : eta = {eta0:.4f}, |w_T| = {np.linalg.norm(path0[-1]):.2e}")
        draw(ax, path, "C0", zorder=3)
        print(f"{label} {algorithm:4s} Tik {gamma:g}  : eta = {eta:.4f}, |w_T| = {np.linalg.norm(path[-1]):.2e}")
        ax.set_title(rf"{label} FB, Tik $\gamma={gamma:g}$" if algorithm == "FB" else f"{label} {algorithm}")
        ax.margins(0.08)
    for ax in axes:
        ax.plot(0, 0, "*", ms=7, color="k", zorder=5)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_xlabel(r"$\theta$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$\varphi$")
    fig.tight_layout(w_pad=0.8)
    path_out = os.path.join(out_dir, "fig_bilinear.pdf")
    fig.savefig(path_out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path_out)


# ----------------------------------------------------------------------------------------------
# Step-size table
# ----------------------------------------------------------------------------------------------
def write_stepsize_table(out_dir, table):
    with open(os.path.join(out_dir, "table_stepsizes.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "Lhat", "eta_FB", "eta_EG", "eta_EFTP"])
        for config, row in table.items():
            writer.writerow([config_tag(config), f"{row['Lhat']:.3f}", f"{row['FB']:.4f}",
                             f"{row['EG']:.4f}", f"{row['EFTP']:.4f}"])

    def line(name, key, digits):
        return name + " & " + " & ".join(f"{table[c][key]:.{digits}f}" for c in CONFIGS) + r"\\"

    tex = [r"\begin{tabular}{l" + "c" * len(CONFIGS) + "}", r"\toprule",
           " & " + " & ".join(PLOT_STYLE[config_tag(c)][0] for c in CONFIGS) + r"\\", r"\midrule",
           line(r"$\hat L$", "Lhat", 2), line(r"$\eta_{\mathrm{FB}}$", "FB", 3),
           line(r"$\eta_{\mathrm{EG}}$", "EG", 3), line(r"$\eta_{\mathrm{EFTP}}$", "EFTP", 3),
           r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out_dir, "table_stepsizes.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")
    for config, row in table.items():
        print(f"{config_tag(config):>5s}: Lhat = {row['Lhat']:.2f}, eta_FB = {row['FB']:.3f}, "
              f"eta_EG = {row['EG']:.3f}, eta_EFTP = {row['EFTP']:.3f}")


# ----------------------------------------------------------------------------------------------
# Multi-start study of the nonlinear example
# ----------------------------------------------------------------------------------------------
def _run_group(task):
    """All starts of one (configuration, algorithm) pair."""
    config, algorithm, eta, starts, n_iter, checkpoints = task
    game, reference = NonlinearGame(*config), NonlinearGame()
    paths = np.zeros((len(starts), 3, len(checkpoints)))
    for i, start in enumerate(starts):
        _, _, paths[i] = run_algorithm(game, reference, start, algorithm, eta, n_iter, checkpoints)
    return config_tag(config), algorithm, paths


def run_multistart(table, n_starts, n_iter, seed, jobs):
    """Every configuration and algorithm from the same random starts."""
    starts = np.random.default_rng(seed).uniform(-BOX, BOX, (n_starts, 3))
    checkpoints = checkpoint_grid(n_iter)
    tasks = [(config, algorithm, table[config][algorithm], starts, n_iter, checkpoints)
             for config in CONFIGS for algorithm in ALGORITHMS]
    paths, t0 = {}, time.time()
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = pool.map(_run_group, tasks)
            for tag, algorithm, group in results:
                paths[(tag, algorithm)] = group
                print(f"{tag:>5s} {algorithm:>4s} done ({time.time() - t0:.0f} s)")
    else:
        for task in tasks:
            tag, algorithm, group = _run_group(task)
            paths[(tag, algorithm)] = group
            print(f"{tag:>5s} {algorithm:>4s} done ({time.time() - t0:.0f} s)")
    return paths, checkpoints


def write_paths(path, paths, checkpoints):
    """One row per run and quantity; column it_K holds the value after K iterations."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["quantity", "config", "alg", "start"] + [f"it_{k}" for k in checkpoints])
        for (tag, algorithm), group in paths.items():
            for i in range(group.shape[0]):
                for q, name in enumerate(QUANTITIES):
                    writer.writerow([name, tag, algorithm, i] + [f"{v:.3e}" for v in group[i, q]])


def read_paths(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        checkpoints = np.array([int(h[3:]) for h in header[4:]])
        runs = {}
        for row in reader:
            runs.setdefault((row[1], row[2]), {}).setdefault(int(row[3]), {})[row[0]] = [float(v) for v in row[4:]]
    paths = {key: np.array([[group[i][q] for q in QUANTITIES] for i in sorted(group)])
             for key, group in runs.items()}
    return paths, checkpoints


def summarize(paths, checkpoints, out_dir, nash_tol=NASH_TOL):
    """Counts and oracle calls per configuration and algorithm."""
    rows = []
    for config in CONFIGS:
        tag = config_tag(config)
        for algorithm in ALGORITHMS:
            if (tag, algorithm) not in paths:
                continue
            group = paths[(tag, algorithm)]
            residual_end, gap = group[:, 0, -1], group[:, 2]
            below = gap <= nash_tol
            first = np.array([oracle_calls(algorithm, checkpoints[np.argmax(b)]) for b in below[below.any(axis=1)]])
            row = {"config": tag, "alg": algorithm, "starts": group.shape[0],
                   "converged": int((residual_end < STOP_TOL).sum()),
                   "nash": int((gap[:, -1] <= nash_tol).sum()),
                   "median_residual_end": f"{np.median(residual_end):.2e}",
                   "median_gap_end": f"{np.median(gap[:, -1]):.2e}",
                   "median_calls_to_nash_tol": int(np.median(first)) if len(first) else "",
                   "max_calls_to_nash_tol": int(first.max()) if len(first) else ""}
            rows.append(row)
            print(f"{tag:>5s} {algorithm:>4s}: converged {row['converged']}/{row['starts']}, "
                  f"Nash {row['nash']}/{row['starts']}, median gap {row['median_gap_end']}, "
                  f"median oracle calls to gap <= {nash_tol:g}: {row['median_calls_to_nash_tol'] or '-'}")
    with open(os.path.join(out_dir, "multistart_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def figure_multistart(paths, checkpoints, out_dir, tags, lo=0.10, hi=0.90, residual_ylim=(1e-4, 30.0),
                      gap_ylim=(1e-4, 10.0), width=6.0, height=3.4, name="fig_multistart"):
    """Residual (top) and Nash gap (bottom) vs oracle calls: median and lo-hi quantiles."""
    fig, axes = plt.subplots(2, len(ALGORITHMS), figsize=(width, height), sharex="col", sharey="row")
    for col, algorithm in enumerate(ALGORITHMS):
        calls = np.maximum(oracle_calls(algorithm, checkpoints), 1)
        for tag in tags:
            if (tag, algorithm) not in paths:
                continue
            _, colour, style = PLOT_STYLE[tag]
            for row, quantity in enumerate((0, 2)):
                values = np.maximum(paths[(tag, algorithm)][:, quantity], PLOT_FLOOR)
                lower, upper = np.quantile(values, lo, axis=0), np.quantile(values, hi, axis=0)
                ax = axes[row, col]
                ax.fill_between(calls, lower, upper, color=colour, alpha=0.18, lw=0)
                ax.plot(calls, lower, color=colour, lw=0.4, alpha=0.6)
                ax.plot(calls, upper, color=colour, lw=0.4, alpha=0.6)
                ax.plot(calls, np.median(values, axis=0), color=colour, ls=style, lw=1.4)
        axes[0, col].axhline(STOP_TOL, color="0.3", lw=0.7, ls="-.")
        axes[1, col].axhline(NASH_TOL, color="#2a7f62", lw=0.7, ls="--")
        axes[0, col].set_title(algorithm)
        axes[1, col].set_xlabel("oracle calls")
        for row in range(2):
            axes[row, col].set_xscale("log")
            axes[row, col].set_yscale("log")
            axes[row, col].grid(alpha=0.3)
        axes[0, col].set_ylim(*residual_ylim)
        axes[1, col].set_ylim(*gap_ylim)
    axes[0, 0].set_ylabel(r"$\|R_{F_\gamma}(\omega_k)\|$")
    axes[1, 0].set_ylabel("Nash gap")
    handles = [Line2D([], [], color=PLOT_STYLE[t][1], ls=PLOT_STYLE[t][2], lw=1.4, label=PLOT_STYLE[t][0])
               for t in tags]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.055, 1, 1), h_pad=0.6)
    path_out = os.path.join(out_dir, name + ".pdf")
    fig.savefig(path_out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path_out)


# ----------------------------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------------------------
def selftest():
    """Gradient, quadrature and Nash-gap checks."""
    rng, ok = np.random.default_rng(0), True
    for penalty in ("none", "tik", "zc"):
        game = NonlinearGame(penalty, 1.0)
        err_loss = err_pen = 0.0
        for _ in range(10):
            omega = rng.uniform(-BOX, BOX, 3)
            fd_loss = np.array([(game.loss(omega + e) - game.loss(omega - e)) / 2e-6 for e in np.eye(3) * 1e-6])
            fd_pen = np.array([(game.penalty_value(omega + e) - game.penalty_value(omega - e)) / 2e-6
                               for e in np.eye(3) * 1e-6])
            err_loss = max(err_loss, np.max(np.abs(fd_loss - game.grad_loss(omega))))
            err_pen = max(err_pen, np.max(np.abs(fd_pen - game.grad_penalty(omega))))
        print(f"{penalty:>4s}: max |grad L - FD| = {err_loss:.1e}, max |grad B - FD| = {err_pen:.1e}")
        ok &= err_loss < 1e-6 and err_pen < 1e-6
    coarse, fine = NonlinearGame("zc", 1.0, 401), NonlinearGame("zc", 1.0, 1601)
    quad_err = max(np.max(np.abs(coarse.operator(w) - fine.operator(w))) for w in rng.uniform(-BOX, BOX, (20, 3)))
    print(f"quadrature, 401 against 1601 nodes: max |F_gamma difference| = {quad_err:.1e}")
    ok &= quad_err < 1e-10
    base = NonlinearGame()
    odd = abs(base.expect(np.tanh(1.3 * base.x_nodes)))
    print(f"E tanh(phi2 x), which is zero by symmetry: {odd:.1e}")
    ok &= odd < 1e-12
    gaps = {p: base.nash_gap(np.array(p)) for p in ((0.0, 1.2, 0.0), (0.0, 0.0, 1.5), (0.5, 0.0, 0.0), (0.5, 1.0, 1.0))}
    for point, gap in gaps.items():
        print(f"Nash gap at {point}: {gap:.2e}")
    ok &= gaps[(0.0, 1.2, 0.0)] < 1e-12 and gaps[(0.0, 0.0, 1.5)] < 1e-12 and gaps[(0.5, 0.0, 0.0)] > 1e-3
    for penalty in ("tik", "zc"):
        res = natural_residual(NonlinearGame(penalty, 1.0).operator, np.array([0.5, 0.0, 0.0]))
        print(f"residual of F_gamma ({penalty}) at (0.5, 0, 0): {res:.1e}")
        ok &= res < 1e-12
    _, eta = bilinear_trajectory("FB", 2.0, np.random.default_rng(7).normal(size=2))
    ok &= abs(eta - 0.3935) < 1e-3
    print("PASSED" if ok else "FAILED")
    return ok


# ----------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["selftest", "bilinear", "stepsizes", "multistart", "plot", "all"])
    parser.add_argument("--out", default="results", help="output folder")
    parser.add_argument("--starts", type=int, default=50, help="number of initial points")
    parser.add_argument("--seed", type=int, default=0, help="seed of the initial points")
    parser.add_argument("--iterations", type=int, default=20000, help="iteration budget per run")
    parser.add_argument("--lipschitz-pairs", type=int, default=4000)
    parser.add_argument("--jobs", type=int, default=1, help="processes for the multi-start runs")
    parser.add_argument("--quick", action="store_true", help="multistart: 4 starts and 1500 iterations")
    parser.add_argument("--csv", default=None, help="plot: paths file (default <out>/multistart_paths.csv)")
    parser.add_argument("--configs", default="none,tik1,tik3,zc1,zc3", help="configurations in the figure")
    parser.add_argument("--residual-ymin", type=float, default=1e-4)
    parser.add_argument("--gap-ymin", type=float, default=1e-4)
    parser.add_argument("--name", default="fig_multistart", help="file name of the multi-start figure")
    args = parser.parse_args()

    if args.command == "selftest":
        sys.exit(0 if selftest() else 1)
    tags = [t.strip() for t in args.configs.split(",") if t.strip()]
    unknown = [t for t in tags if t not in PLOT_STYLE]
    if unknown:
        parser.error(f"unknown --configs entries {unknown}; choose from {', '.join(PLOT_STYLE)}")
    if args.quick:
        args.starts, args.iterations, args.lipschitz_pairs = 4, 1500, 400
        args.out += "_quick"
    os.makedirs(args.out, exist_ok=True)
    limits = {"residual_ylim": (args.residual_ymin, 30.0), "gap_ylim": (args.gap_ymin, 10.0), "name": args.name}

    if args.command in ("bilinear", "all"):
        figure_bilinear(args.out)
    table = None
    if args.command in ("stepsizes", "multistart", "all"):
        table = step_sizes(args.lipschitz_pairs)
        write_stepsize_table(args.out, table)
    if args.command in ("multistart", "all"):
        paths, checkpoints = run_multistart(table, args.starts, args.iterations, args.seed, args.jobs)
        write_paths(os.path.join(args.out, "multistart_paths.csv"), paths, checkpoints)
        summarize(paths, checkpoints, args.out)
        figure_multistart(paths, checkpoints, args.out, tags, **limits)
    if args.command == "plot":
        csv_path = args.csv or os.path.join(args.out, "multistart_paths.csv")
        if not os.path.exists(csv_path):
            parser.error(f"{csv_path} not found; run 'python main.py multistart' first or pass --csv")
        paths, checkpoints = read_paths(csv_path)
        summarize(paths, checkpoints, args.out)
        figure_multistart(paths, checkpoints, args.out, tags, **limits)


if __name__ == "__main__":
    main()
