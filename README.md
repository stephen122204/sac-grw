# Historical RB-GBMC studies

This branch preserves the earlier relaxation–Brownian gradient-particle studies
for Burgers, heat, and FitzHugh–Nagumo equations. The current SAC-GRW coupling
paper and code are on [main](https://github.com/stephen122204/sac-grw).

## Install and verify

```bash
git clone --branch rb-gbmc-paper2 https://github.com/stephen122204/sac-grw.git
cd sac-grw
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py verify
```

The `rb-gbmc-paper2` and `legacy-paper2-pre-split` branches preserve the same
historical snapshot. Verification reads saved results without rerunning
simulations. Configurations and tolerances are recorded in `reproduce.py`
and `expected_values.json`.

Use `python reproduce.py --help` for study targets. Full reruns can overwrite
stored outputs, so use a separate copy. Figures can be regenerated with
`python reproduce.py figures`. These historical results do not establish the
claims of the current coupling paper.
