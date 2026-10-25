//! The training loop from `demo.ipynb`: max-margin loss with L2 regularization,
//! optimized with plain SGD and a linearly decaying learning rate.
//!
//! Each expression is written in the notebook's order (for example `* (1.0 / n)`
//! rather than `/ n`) so the results match Python bit for bit.

use crate::engine::Value;
use crate::nn::{MLP, Module};

/// Total loss over the whole dataset and the fraction classified correctly.
/// `ys` are the labels -1.0 or 1.0; the model must have a single output.
pub fn loss(model: &MLP, xs: &[Vec<f64>], ys: &[f64], alpha: f64) -> (Value, f64) {
    let scores: Vec<Value> = xs
        .iter()
        .map(|row| {
            let input: Vec<Value> = row.iter().copied().map(Value::new).collect();
            model.forward(&input)[0].clone()
        })
        .collect();

    // svm "max-margin" loss
    let losses: Vec<Value> = ys
        .iter()
        .zip(&scores)
        .map(|(&yi, score)| (1.0 + -yi * score).relu())
        .collect();
    let n = losses.len();
    let data_loss = losses.into_iter().sum::<Value>() * (1.0 / n as f64);

    // L2 regularization
    let reg_loss = alpha * model.parameters().iter().map(|p| p * p).sum::<Value>();
    let total_loss = data_loss + reg_loss;

    let correct = ys
        .iter()
        .zip(&scores)
        .filter(|&(&yi, score)| (yi > 0.0) == (score.data() > 0.0))
        .count();
    (total_loss, correct as f64 / ys.len() as f64)
}

/// The notebook's schedule: 1.0 at step 0, decaying linearly towards 0.1.
pub fn learning_rate(k: usize, steps: usize) -> f64 {
    1.0 - 0.9 * k as f64 / steps as f64
}

/// One optimization step: forward, backward, SGD update. Returns the loss and
/// accuracy measured before the update, like the notebook prints them.
pub fn step(model: &MLP, xs: &[Vec<f64>], ys: &[f64], alpha: f64, lr: f64) -> (f64, f64) {
    let (total_loss, accuracy) = loss(model, xs, ys, alpha);
    model.zero_grad();
    total_loss.backward();
    for p in model.parameters() {
        p.set_data(p.data() - lr * p.grad());
    }
    (total_loss.data(), accuracy)
}
