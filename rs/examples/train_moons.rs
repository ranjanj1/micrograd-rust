//! demo.ipynb's training loop, for benchmarking against Python and exporting the
//! decision boundary for animation.
//!
//! Takes the dataset, layer sizes and initial weights from a JSON file in
//! golden.json's format (tests/golden.json by default, or one written by
//! scripts/export_moons.py), since Rust can't reproduce numpy's and Python's
//! random generators.
//!
//!     cargo run --release --example train_moons -- [DATA.json] [--boundary OUT.json]
//!
//! With `--boundary`, the model is scored on the notebook's grid (step 0.25, one
//! unit past the data on each side) before training and after every step.

#![forbid(unsafe_code)]

use std::time::{Duration, Instant};

use micrograd::train::{learning_rate, step};
use micrograd::{MLP, Module, Value};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Data {
    config: Config,
    #[serde(rename = "X")]
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    initial_weights: Vec<Vec<NeuronWeights>>,
}

#[derive(Deserialize)]
struct Config {
    nin: usize,
    nouts: Vec<usize>,
    alpha: f64,
    steps: usize,
}

#[derive(Deserialize)]
struct NeuronWeights {
    w: Vec<f64>,
}

/// Everything an animation needs. `frames[k]` is the model after `k` training
/// steps, so `frames[0]` is the untrained model and `steps[k]` is the loss and
/// accuracy measured on `frames[k]` (before step `k`'s update).
#[derive(Serialize)]
struct Boundary<'a> {
    xs: Vec<f64>,
    ys: Vec<f64>,
    points: &'a [Vec<f64>],
    labels: &'a [f64],
    steps: Vec<StepLog>,
    frames: Vec<Frame>,
}

#[derive(Serialize)]
struct StepLog {
    step: usize,
    loss: f64,
    accuracy: f64,
}

#[derive(Serialize)]
struct Frame {
    updates: usize,
    /// Raw model score at each grid point, `scores[row][col]` for `(xs[col], ys[row])`;
    /// the decision boundary is where it crosses 0.
    scores: Vec<Vec<f64>>,
}

const GRID_STEP: f64 = 0.25;

fn main() {
    let mut data_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden.json").to_string();
    let mut boundary_path = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--boundary" {
            boundary_path = Some(args.next().expect("--boundary needs a path"));
        } else {
            data_path = arg;
        }
    }

    let text = std::fs::read_to_string(&data_path).expect("read data file");
    let d: Data = serde_json::from_str(&text).expect("parse data file");
    let c = &d.config;

    let mut draws = d
        .initial_weights
        .iter()
        .flatten()
        .flat_map(|n| n.w.iter().copied());
    let model = MLP::new(c.nin, &c.nouts, &mut || {
        draws.next().expect("initial weight")
    });
    println!(
        "{} parameters, {} points",
        model.parameters().len(),
        d.x.len()
    );

    let (xs, ys) = (grid_axis(&d.x, 0), grid_axis(&d.x, 1));
    let mut frames = Vec::new();
    let mut scoring = Duration::ZERO;
    let mut score_grid = |updates: usize, frames: &mut Vec<Frame>| {
        let start = Instant::now();
        let scores = ys
            .iter()
            .map(|&y| xs.iter().map(|&x| predict(&model, x, y)).collect())
            .collect();
        frames.push(Frame { updates, scores });
        scoring += start.elapsed();
    };

    if boundary_path.is_some() {
        score_grid(0, &mut frames);
    }
    let mut steps = Vec::with_capacity(c.steps);
    let mut training = Duration::ZERO;
    for k in 0..c.steps {
        let start = Instant::now();
        let (loss, accuracy) = step(&model, &d.x, &d.y, c.alpha, learning_rate(k, c.steps));
        training += start.elapsed();
        if k < 3 || k == c.steps - 1 {
            println!("step {k} loss {loss}, accuracy {}%", accuracy * 100.0);
        }
        steps.push(StepLog {
            step: k,
            loss,
            accuracy,
        });
        if boundary_path.is_some() {
            score_grid(k + 1, &mut frames);
        }
    }
    println!(
        "trained {} steps in {:.3} s ({:.1} ms/step)",
        c.steps,
        training.as_secs_f64(),
        training.as_secs_f64() * 1e3 / c.steps as f64
    );

    if let Some(path) = boundary_path {
        let n_frames = frames.len();
        let boundary = Boundary {
            xs,
            ys,
            points: &d.x,
            labels: &d.y,
            steps,
            frames,
        };
        let json = serde_json::to_string(&boundary).expect("serialize boundary");
        std::fs::write(&path, json).expect("write boundary file");
        println!(
            "wrote {path}: {n_frames} frames of {}x{} grid, scored in {:.3} s",
            boundary.ys.len(),
            boundary.xs.len(),
            scoring.as_secs_f64()
        );
    }
}

fn predict(model: &MLP, x: f64, y: f64) -> f64 {
    model.forward(&[Value::new(x), Value::new(y)])[0].data()
}

/// The notebook's `np.arange(min - 1, max + 1, h)` over one input column.
fn grid_axis(points: &[Vec<f64>], col: usize) -> Vec<f64> {
    let values = points.iter().map(|p| p[col]);
    let lo = values.clone().fold(f64::INFINITY, f64::min) - 1.0;
    let hi = values.fold(f64::NEG_INFINITY, f64::max) + 1.0;
    let n = ((hi - lo) / GRID_STEP).ceil() as usize;
    (0..n).map(|i| lo + i as f64 * GRID_STEP).collect()
}
