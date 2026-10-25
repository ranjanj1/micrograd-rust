"""Export a moons dataset and initial MLP weights in golden.json's format, so the
Rust port can train the same model the Python code would (no training is run).

Seeds, data and initialization follow demo.ipynb; only the sizes change.
Run from the repo root:
    ../.venv/bin/python rs/scripts/export_moons.py 1000 32 32 1
"""
import json
import os
import random
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

import numpy as np
from sklearn.datasets import make_moons

from micrograd.nn import MLP


def main():
    n_samples = int(sys.argv[1])
    nouts = [int(a) for a in sys.argv[2:]]

    np.random.seed(1337)
    random.seed(1337)
    X, y = make_moons(n_samples=n_samples, noise=0.1)
    y = y * 2 - 1
    model = MLP(2, nouts)

    data = {
        "config": {"seed": 1337, "n_samples": n_samples, "noise": 0.1, "nin": 2,
                   "nouts": nouts, "alpha": 1e-4, "steps": 100},
        "X": [[float(a), float(b)] for a, b in X],
        "y": [int(v) for v in y],
        "initial_weights": [
            [{"w": [float(w.data) for w in n.w], "b": float(n.b.data)} for n in layer.neurons]
            for layer in model.layers
        ],
    }
    name = f"moons_{n_samples}_{'x'.join(map(str, nouts))}.json"
    out = os.path.join(REPO, "rs", "data", name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=1)
    print(f"wrote {out}: {n_samples} points, {len(model.parameters())} parameters")


if __name__ == "__main__":
    main()
