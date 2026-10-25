# CLAUDE.md

We are porting micrograd (a tiny scalar autograd engine and neural net library) from Python to Rust.

## Layout

- `micrograd/engine.py`, `micrograd/nn.py`, `test/test_engine.py`: the original Python reference implementation.
- `rs/`: the new Rust cargo crate that holds the port.

## Rules

- **Never edit the Python files.** They are the reference implementation. Read them, run them, compare against them, but do not modify them (this includes `micrograd/`, `test/`, `setup.py` and the notebooks).
- All Rust code lives in the `rs/` crate.
- The crate root (`rs/src/lib.rs`, and any binary roots) must have `#![forbid(unsafe_code)]`.
- After every change, run both from `rs/` and fix any failures or warnings before moving on:
  ```sh
  cargo test
  cargo clippy
  ```
- Work in small steps. For each step, explain each design choice in one or two sentences.

## Python environment

Python with numpy, scikit-learn and torch is at `../.venv/bin/python`. Use it to run the Python reference and tests (for example, `../.venv/bin/python -m pytest test/`) and to generate expected values for checking the Rust port against.
