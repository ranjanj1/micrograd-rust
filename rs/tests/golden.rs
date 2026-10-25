//! Replays demo.ipynb's 100 training steps and checks them bit for bit against
//! tests/golden.json, written by scripts/export_golden.py from the Python code.

#![forbid(unsafe_code)]

use micrograd::train::{learning_rate, step};
use micrograd::{MLP, Module, Value};
use serde::Deserialize;

#[derive(Deserialize)]
struct Golden {
    config: Config,
    #[serde(rename = "X")]
    x: Vec<Vec<f64>>,
    y: Vec<i64>,
    initial_weights: Vec<Vec<NeuronWeights>>,
    steps: Vec<Step>,
    final_weights: Vec<Vec<NeuronWeights>>,
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
    b: f64,
}

#[derive(Deserialize)]
struct Step {
    step: usize,
    loss: f64,
    accuracy: f64,
    learning_rate: f64,
}

fn load() -> Golden {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden.json");
    let text = std::fs::read_to_string(path).expect("read golden.json");
    serde_json::from_str(&text).expect("parse golden.json")
}

/// Weights flattened in `parameters()` order: [w0, w1, ..., b] per neuron.
fn flatten(layers: &[Vec<NeuronWeights>]) -> Vec<f64> {
    layers
        .iter()
        .flatten()
        .flat_map(|n| n.w.iter().copied().chain([n.b]))
        .collect()
}

fn param_data(model: &MLP) -> Vec<f64> {
    model.parameters().iter().map(Value::data).collect()
}

#[test]
fn training_matches_python_bit_for_bit() {
    let g = load();
    assert_eq!(g.steps.len(), g.config.steps);

    // Python draws one random.uniform per weight in creation order; biases start at 0.
    let mut draws = g
        .initial_weights
        .iter()
        .flatten()
        .flat_map(|n| n.w.iter().copied());
    let model = MLP::new(g.config.nin, &g.config.nouts, &mut || {
        draws.next().expect("ran out of initial weights")
    });
    assert_eq!(draws.next(), None, "golden.json has unused initial weights");
    assert_eq!(param_data(&model), flatten(&g.initial_weights));

    // Compare everything before failing, so the report shows the worst difference
    // and every mismatch rather than stopping at the first one.
    let mut lr_diff = Diff::new("learning rate");
    let mut loss_diff = Diff::new("loss");
    let mut acc_diff = Diff::new("accuracy");
    let ys: Vec<f64> = g.y.iter().map(|&v| v as f64).collect();
    for expected in &g.steps {
        let k = expected.step;
        let lr = learning_rate(k, g.config.steps);
        lr_diff.check(format!("step {k}"), lr, expected.learning_rate);
        let (loss, accuracy) = step(&model, &g.x, &ys, g.config.alpha, lr);
        loss_diff.check(format!("step {k}"), loss, expected.loss);
        acc_diff.check(format!("step {k}"), accuracy, expected.accuracy);
    }

    let mut weight_diff = Diff::new("final weight");
    for (i, (got, want)) in param_data(&model)
        .into_iter()
        .zip(flatten(&g.final_weights))
        .enumerate()
    {
        weight_diff.check(format!("parameter {i}"), got, want);
    }

    let diffs = [lr_diff, loss_diff, acc_diff, weight_diff];
    for d in &diffs {
        println!("{}", d.summary());
    }
    let mismatches: Vec<&String> = diffs.iter().flat_map(|d| &d.mismatches).collect();
    assert!(
        mismatches.is_empty(),
        "{} values differ from Python, first: {}",
        mismatches.len(),
        mismatches[0]
    );
}

/// Tracks the largest gap between Rust and Python values of one kind.
struct Diff {
    kind: &'static str,
    count: usize,
    worst: Option<(String, f64, f64)>,
    mismatches: Vec<String>,
}

impl Diff {
    fn new(kind: &'static str) -> Self {
        Diff {
            kind,
            count: 0,
            worst: None,
            mismatches: Vec::new(),
        }
    }

    fn check(&mut self, at: String, got: f64, want: f64) {
        self.count += 1;
        if got.to_bits() != want.to_bits() {
            self.mismatches.push(format!(
                "{} at {at}: rust {got:?}, python {want:?}",
                self.kind
            ));
        }
        let gap = (got - want).abs();
        if self
            .worst
            .as_ref()
            .is_none_or(|(_, g, w)| gap > (g - w).abs())
        {
            self.worst = Some((at, got, want));
        }
    }

    fn summary(&self) -> String {
        let (at, got, want) = self.worst.as_ref().expect("no values checked");
        format!(
            "{:>13}: {:>3} values, {} differ; worst |rust - python| = {:e} at {at} \
             (rust {got:?}, python {want:?}, {} ulp)",
            self.kind,
            self.count,
            self.mismatches.len(),
            (got - want).abs(),
            ulps(*got, *want),
        )
    }
}

/// Distance in units in the last place; 0 means identical bits.
fn ulps(a: f64, b: f64) -> u64 {
    let key = |x: f64| {
        let bits = x.to_bits() as i64;
        if bits < 0 { i64::MIN - bits } else { bits }
    };
    key(a).abs_diff(key(b))
}
