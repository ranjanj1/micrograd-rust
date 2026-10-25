"""Export golden data from the Python micrograd demo for the Rust port to match.

Reproduces demo.ipynb (seed, make_moons, MLP(2, [16, 16, 1]), hinge loss + L2,
100 SGD steps) and writes rs/tests/golden.json with the dataset, the initial
weights, loss/accuracy at every step, and the final weights.

Run from the repo root:
    ../.venv/bin/python rs/scripts/export_golden.py
"""
import json
import os
import random
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

import numpy as np
from sklearn.datasets import make_moons

from micrograd.engine import Value
from micrograd.nn import MLP

# Value stores its children in a set, so the backward pass order (and with it the
# float summation order of gradients) depends on object addresses and changes from
# run to run. Pin it to insertion order (self, other), deduplicated, which is what
# the Rust port will use. This patches the class in this process only.
_orig_init = Value.__init__


def _ordered_init(self, data, _children=(), _op=""):
    _orig_init(self, data, (), _op)
    self._prev = tuple(dict.fromkeys(_children))


Value.__init__ = _ordered_init


def weights(model):
    return [
        [{"w": [float(w.data) for w in n.w], "b": float(n.b.data)} for n in layer.neurons]
        for layer in model.layers
    ]


def main():
    np.random.seed(1337)
    random.seed(1337)
    X, y = make_moons(n_samples=100, noise=0.1)
    y = y * 2 - 1

    model = MLP(2, [16, 16, 1])
    initial = weights(model)

    def loss():
        inputs = [list(map(Value, xrow)) for xrow in X]
        scores = list(map(model, inputs))
        losses = [(1 + -yi * scorei).relu() for yi, scorei in zip(y, scores)]
        data_loss = sum(losses) * (1.0 / len(losses))
        alpha = 1e-4
        reg_loss = alpha * sum((p * p for p in model.parameters()))
        total_loss = data_loss + reg_loss
        accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(y, scores)]
        return total_loss, sum(accuracy) / len(accuracy)

    steps = []
    for k in range(100):
        total_loss, acc = loss()
        model.zero_grad()
        total_loss.backward()
        learning_rate = 1.0 - 0.9 * k / 100
        for p in model.parameters():
            p.data -= learning_rate * p.grad
        steps.append({"step": k, "loss": float(total_loss.data), "accuracy": float(acc),
                      "learning_rate": learning_rate})

    golden = {
        "config": {"seed": 1337, "n_samples": 100, "noise": 0.1, "nin": 2,
                   "nouts": [16, 16, 1], "alpha": 1e-4, "steps": 100},
        "X": [[float(a), float(b)] for a, b in X],
        "y": [int(v) for v in y],
        "initial_weights": initial,
        "steps": steps,
        "final_weights": weights(model),
    }
    out = os.path.join(REPO, "rs", "tests", "golden.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(golden, f, indent=1)
    print(f"wrote {out}: final loss {steps[-1]['loss']}, accuracy {steps[-1]['accuracy']}")


if __name__ == "__main__":
    main()
