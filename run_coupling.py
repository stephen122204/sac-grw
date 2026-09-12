"""Run a coupled field estimate with user-selected parameters.

This is an exploratory example, not a paper benchmark or a tolerance controller.
The initial cumulative field is a deterministic step approximation to the chosen
profile on [-3, 3], continued constantly outside that interval.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from coupled_gradient_particles import advance_pair
from relaxation_gbmc import reconstruct_cumulative_field


def run(args):
    if args.particles < 2 or args.particles % 2:
        raise ValueError("particles must be a positive even integer of at least 2")
    if args.pairs < 2:
        raise ValueError("at least two independent pairs are needed to estimate variance")
    if not np.isfinite([args.nu, args.dt, args.time]).all():
        raise ValueError("nu, dt and time must be finite")
    if args.nu < 0 or args.dt <= 0 or args.time <= 0:
        raise ValueError("nu must be nonnegative; dt and time must be positive")
    steps = round(args.time / args.dt)
    if steps < 1 or not np.isclose(steps * args.dt, args.time, rtol=1e-12, atol=0):
        raise ValueError("time must be an integer multiple of dt")
    edges = np.linspace(-3.0, 3.0, args.particles + 1)
    values = (np.exp(-edges**2) if args.profile == "gaussian"
              else -np.tanh(edges))
    initial = ((edges[:-1] + edges[1:]) / 2, np.diff(values), float(values[0]))
    derivatives = {"heat": lambda u: np.zeros_like(u),
                   "burgers": lambda u: u, "cubic": lambda u: u**2}
    grid = np.linspace(-6.0, 6.0, 801)
    fields = []
    start = time.perf_counter()
    for seed in np.random.SeedSequence(args.seed).spawn(args.pairs):
        A, B, left = advance_pair(
            initial, args.nu, args.dt, steps, np.random.default_rng(seed),
            args.policy, derivatives[args.flux])
        fields.append(0.5 * (reconstruct_cumulative_field(*A, left, grid)
                             + reconstruct_cumulative_field(*B, left, grid)))
    elapsed = time.perf_counter() - start
    fields = np.asarray(fields)
    variance = fields.var(axis=0, ddof=1)
    dx = grid[1] - grid[0]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "fields.npz", x=grid, paired_fields=fields,
                        mean=fields.mean(axis=0), standard_error=np.sqrt(variance / args.pairs))
    summary = vars(args).copy()
    summary.update(steps=steps, trajectories=2 * args.pairs,
                   integrated_pair_variance=float(dx * variance.sum()),
                   integrated_mean_variance_estimate=float(dx * variance.sum() / args.pairs),
                   simulation_and_reconstruction_seconds=elapsed,
                   note="Sampling variance on [-6,6]; no reference error or bias estimate. "
                        "Initial step field samples the chosen profile on [-3,3]. "
                        "Timing excludes output writes.")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {output / 'fields.npz'} and {output / 'summary.json'}")
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("gaussian", "shock"), default="gaussian")
    parser.add_argument("--flux", choices=("heat", "burgers", "cubic"), default="burgers")
    parser.add_argument("--policy", choices=("FULL", "RAW", "WITHIN", "FINAL"), default="FULL")
    parser.add_argument("--particles", type=int, default=400)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="output/custom/example")
    args = parser.parse_args()
    try:
        run(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
