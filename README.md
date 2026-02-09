# GRW Heat Solver (JSON-driven)

This project simulates the **1D heat equation** using a **Gaussian Random Walk (GRW)** / Monte Carlo particle method.  
Runs are configured via a single **JSON file** (domain, boundary conditions, diffusion constant, timestep, initial condition, etc.) and the script automatically **simulates + plots** results.

---

## Quick Start

### 1) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

### 2) Install dependencies
```bash
pip install numpy matplotlib
```

### 3) Run a simulation from a JSON config
```bash
python main.py configs/heat_gaussian_neumann.json
```
Outputs (PNG + any text outputs) are saved under output/.

---

## JSON Configuration Format
*A config file specifies:*
- the equation type ("heat")
- the domain ("Finite" + domain_size)
- boundary conditions (LEFT/RIGHT, Dirichlet or Neumann)
- diffusion constant diff_constant (α)
- time_step and total_time
- number of particles num_points
- initial condition (e.g., Gaussian cloud)


## Minimal example (Gaussian + Neumann–Neumann)
Save as json_tests/heat_gaussian_neumann.json:
```bash
{
  "equation_type": "heat",
  "domain_type": "Finite",
  "domain_size": 10,

  "boundary_conditions": {
    "LEFT":  { "type": "Neumann", "value": 0 },
    "RIGHT": { "type": "Neumann", "value": 0 }
  },

  "diff_constant": 0.1,
  "time_step": 0.001,
  "total_time": 0.2,
  "num_points": 80000,
  "reaction_term": false,

  "heat_initial_condition": {
    "type": "gaussian_cloud",
    "center": 2.0,
    "sigma": 0.4
  }
}
```
Run:
```bash
python main.py json_tests/heat_gaussian_neumann.json
```
