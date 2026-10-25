# Reproduces demo.ipynb cells 3-7 (no plotting) and times the 100-step training loop.
import sys, time, random
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.datasets import make_moons
from micrograd.engine import Value
from micrograd.nn import MLP

np.random.seed(1337); random.seed(1337)
X, y = make_moons(n_samples=100, noise=0.1)
y = y*2 - 1
model = MLP(2, [16, 16, 1])

def loss():
    inputs = [list(map(Value, xrow)) for xrow in X]
    scores = list(map(model, inputs))
    losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(y, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    reg_loss = 1e-4 * sum((p*p for p in model.parameters()))
    total_loss = data_loss + reg_loss
    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(y, scores)]
    return total_loss, sum(accuracy) / len(accuracy)

print("params", len(model.parameters()))
t0 = time.perf_counter()
for k in range(100):
    total_loss, acc = loss()
    model.zero_grad()
    total_loss.backward()
    lr = 1.0 - 0.9*k/100
    for p in model.parameters():
        p.data -= lr * p.grad
    if k in (0, 99):
        print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")
dt = time.perf_counter() - t0
print(f"100 steps: {dt:.2f} s ({dt*10:.1f} ms/step)")
