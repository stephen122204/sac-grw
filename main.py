# main.py
import sys
import config as config_module
from utils import plot_results

from simulation import (
    simulate_heat_equation,
    simulate_burgers,
    simulate_fitzhugh_nagumo,
)


def main():
    # Usage:
    #   python main.py                 -> interactive prompts
    #   python main.py burgers.json    -> load config from JSON file
    if len(sys.argv) >= 2:
        cfg_path = sys.argv[1]
        cfg = config_module.load_config_from_json(cfg_path)
        print(f"[main] Loaded config from file: {cfg_path}")
    else:
        cfg = config_module.get_user_input()

    eq = (cfg.equation_type or "").strip().lower()

    if eq == "heat":
        initial_globs = [{'position': pos, 'value': val} for pos, val in cfg.initial_conditions]
        results = simulate_heat_equation(initial_globs, cfg)
        print("Simulation Complete! Plotting Results...")
        plot_results(results, cfg.equation_type, cfg)
        return

    if eq in {"fitzhugh-nagumo", "fitzhugh–nagumo", "fitzhugh", "nagumo"}:
        initial_globs = [{'position': pos, 'value': list(val)} for pos, val in cfg.initial_conditions]
        results = simulate_fitzhugh_nagumo(initial_globs, cfg)
        print("Simulation Complete! Plotting Results...")
        plot_results(results, cfg.equation_type, cfg)
        return

    if eq in {"burgers", "burger", "burgers'"}:
        # IMPORTANT for this repo: burgers solver expects value stored like [u]
        initial_globs = [{'position': pos, 'value': [float(val)]} for pos, val in cfg.initial_conditions]
        results = simulate_burgers(initial_globs, cfg)
        print("Simulation Complete! Plotting Results...")
        plot_results(results, cfg.equation_type, cfg)
        return

    raise ValueError(f"Unknown equation type: {cfg.equation_type!r}")


if __name__ == "__main__":
    main()