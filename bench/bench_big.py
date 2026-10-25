# Times the first few training steps of MLP(2, [32, 32, 1]) on 1000 moons points
# with the unmodified Python micrograd, same seeds as export_moons.py.
import sys, time, random
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.datasets import make_moons
from micrograd.engine import Value
from micrograd.nn import MLP

# micrograd's recursive build_topo goes deeper than the default limit of 1000
sys.setrecursionlimit(10_000)
N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
np.random.seed(1337); random.seed(1337)
X, y = make_moons(n_samples=1000, noise=0.1)
y = y*2 - 1
model = MLP(2, [32, 32, 1])
print("params", len(model.parameters()), "points", len(X))

def loss():
    inputs = [list(map(Value, xrow)) for xrow in X]
    scores = list(map(model, inputs))
    losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(y, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    reg_loss = 1e-4 * sum((p*p for p in model.parameters()))
    total_loss = data_loss + reg_loss
    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(y, scores)]
    return total_loss, sum(accuracy) / len(accuracy)

times = []
for k in range(N_STEPS):
    t0 = time.perf_counter()
    total_loss, acc = loss()
    model.zero_grad()
    total_loss.backward()
    lr = 1.0 - 0.9*k/100
    for p in model.parameters():
        p.data -= lr * p.grad
    times.append(time.perf_counter() - t0)
    print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%  ({times[-1]:.2f} s)", flush=True)
print("per-step mean %.2f s, min %.2f s, max %.2f s" % (sum(times)/len(times), min(times), max(times)))
