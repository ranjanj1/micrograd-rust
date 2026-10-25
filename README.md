# micrograd in Rust (ported with an AI agent)

A Rust port of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) (MIT), made on camera with
Claude Code for the It's Not Magic video "Why Is AI Rewriting Code in Rust?". The Python original is included
unchanged as the reference.

## Results (Apple M1, from the recorded session)
- All 637 values (every loss, accuracy and final weight over 100 training steps) match the Python bit for bit (0 ulp).
- Demo, 100 steps: Python 69.3 s, Rust 0.51 s (about 137×), measured with hyperfine (`bench/*.json`).
- Bigger network, 1,000 points: Rust 22.2 s. Python hit a RecursionError; its run is estimated at ~95 min from
  6 measured steps (`bench/bench_big_6.log`).
- `unsafe` code: 0 lines (`#![forbid(unsafe_code)]`).
- Design: shared handles (`Rc<RefCell<…>>`), a one-to-one port of the Python. A flat "tape" design is faster still;
  part of the speed is the language and part is the design.

## Layout
- `micrograd/`, `test/`: the original Python (reference, not edited).
- `rs/`: the Rust crate (`src/engine.rs`, `src/nn.rs`, `src/train.rs`, `examples/train_moons.rs`).
- `rs/tests/golden.rs` + `golden.json`: every number from the Python run, checked against the Rust port.
- `rs/scripts/`: exporters that produce the golden data from the Python.
- `bench/`: the benchmark scripts and hyperfine results.
- `CLAUDE.md` and `prompts.txt`: the rules and the exact prompts used in the session.

## Run
```sh
cd rs
cargo test
cargo run --release --example train_moons
```
Python reference (needs numpy and scikit-learn): `python bench/bench_demo.py`.

## Licence
MIT. Original micrograd © 2020 Andrej Karpathy; the Rust port © 2026 Ranjan Jaiswal. See LICENSE.
