//! Neurons, layers and MLPs built from `Value`, ported from `micrograd/nn.py`.
//!
//! Constructors take an `init` closure for the weights instead of calling a
//! random generator: Python draws `random.uniform(-1, 1)` once per weight in
//! creation order, so passing those draws back in reproduces the Python model.

use std::fmt;

use crate::engine::Value;

pub trait Module {
    /// All trainable values, in the same order as Python's `parameters()`.
    fn parameters(&self) -> Vec<Value>;

    fn zero_grad(&self) {
        for p in self.parameters() {
            p.set_grad(0.0);
        }
    }
}

pub struct Neuron {
    pub w: Vec<Value>,
    pub b: Value,
    pub nonlin: bool,
}

impl Neuron {
    pub fn new(nin: usize, nonlin: bool, init: &mut impl FnMut() -> f64) -> Self {
        let w = (0..nin).map(|_| Value::new(init())).collect();
        Neuron {
            w,
            b: Value::new(0.0),
            nonlin,
        }
    }

    /// `relu(b + w0*x0 + w1*x1 + ...)`, summed left to right from the bias like
    /// Python's `sum(..., self.b)`.
    pub fn forward(&self, x: &[Value]) -> Value {
        let act = self
            .w
            .iter()
            .zip(x)
            .fold(self.b.clone(), |acc, (wi, xi)| acc + wi * xi);
        if self.nonlin { act.relu() } else { act }
    }
}

impl Module for Neuron {
    fn parameters(&self) -> Vec<Value> {
        let mut params = self.w.clone();
        params.push(self.b.clone());
        params
    }
}

impl fmt::Display for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = if self.nonlin { "ReLU" } else { "Linear" };
        write!(f, "{kind}Neuron({})", self.w.len())
    }
}

pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize, nonlin: bool, init: &mut impl FnMut() -> f64) -> Self {
        let neurons = (0..nout).map(|_| Neuron::new(nin, nonlin, init)).collect();
        Layer { neurons }
    }

    /// One output per neuron. Unlike Python, a single-neuron layer still returns
    /// a one-element `Vec`, so the return type stays fixed.
    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(x)).collect()
    }
}

impl Module for Layer {
    fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(Neuron::parameters).collect()
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Layer of [{}]", join(&self.neurons))
    }
}

pub struct MLP {
    pub layers: Vec<Layer>,
}

impl MLP {
    /// ReLU on every layer except the last, which is linear.
    pub fn new(nin: usize, nouts: &[usize], init: &mut impl FnMut() -> f64) -> Self {
        // Each layer's input size is the previous layer's output size.
        let nins = std::iter::once(nin).chain(nouts.iter().copied());
        let layers = nins
            .zip(nouts)
            .enumerate()
            .map(|(i, (nin, &nout))| Layer::new(nin, nout, i + 1 < nouts.len(), init))
            .collect();
        MLP { layers }
    }

    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        let mut x = x.to_vec();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }
}

impl Module for MLP {
    fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(Layer::parameters).collect()
    }
}

impl fmt::Display for MLP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MLP of [{}]", join(&self.layers))
    }
}

fn join(items: &[impl fmt::Display]) -> String {
    items
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hands out the given values in order, standing in for Python's random draws.
    fn from_list(values: &[f64]) -> impl FnMut() -> f64 + '_ {
        let mut it = values.iter().copied();
        move || it.next().expect("ran out of init values")
    }

    fn counter() -> impl FnMut() -> f64 {
        let mut n = 0.0;
        move || {
            n += 1.0;
            n
        }
    }

    const SMALL_INIT: [f64; 9] = [0.5, -0.3, 0.8, 0.1, -0.7, 0.2, 0.9, -0.4, 0.6];

    #[test]
    fn display_matches_python_repr() {
        let m = MLP::new(2, &[2, 1], &mut counter());
        assert_eq!(
            m.to_string(),
            "MLP of [Layer of [ReLUNeuron(2), ReLUNeuron(2)], Layer of [LinearNeuron(2)]]"
        );
    }

    #[test]
    fn demo_model_has_337_parameters() {
        let m = MLP::new(2, &[16, 16, 1], &mut counter());
        assert_eq!(m.parameters().len(), 337);
    }

    // Python: [w0, w1, b] per neuron, neuron by neuron, layer by layer, with weights
    // drawn in that same order and biases starting at 0.
    #[test]
    fn parameter_order_matches_python() {
        let m = MLP::new(2, &[3, 1], &mut from_list(&SMALL_INIT));
        let data: Vec<f64> = m.parameters().iter().map(Value::data).collect();
        assert_eq!(
            data,
            [
                0.5, -0.3, 0.0, 0.8, 0.1, 0.0, -0.7, 0.2, 0.0, 0.9, -0.4, 0.6, 0.0
            ]
        );
    }

    // Expected values from the Python reference with the same weights and input.
    #[test]
    fn forward_and_backward_match_python() {
        let m = MLP::new(2, &[3, 1], &mut from_list(&SMALL_INIT));
        let out = m.forward(&[Value::new(1.5), Value::new(-2.0)]);
        assert_eq!(out.len(), 1);
        out[0].backward();
        assert_eq!(out[0].data(), 0.815);
        let grads: Vec<f64> = m.parameters().iter().map(Value::grad).collect();
        assert_eq!(
            grads,
            [
                1.35,
                -1.8,
                0.9,
                -0.6000000000000001,
                0.8,
                -0.4,
                0.0,
                0.0,
                0.0,
                1.35,
                1.0000000000000002,
                0.0,
                1.0
            ]
        );
    }

    #[test]
    fn parameters_are_shared_handles() {
        let m = MLP::new(2, &[3, 1], &mut from_list(&SMALL_INIT));
        m.parameters()[0].set_data(42.0);
        assert_eq!(m.layers[0].neurons[0].w[0].data(), 42.0);
    }

    #[test]
    fn zero_grad_clears_all_grads() {
        let m = MLP::new(2, &[3, 1], &mut from_list(&SMALL_INIT));
        m.forward(&[Value::new(1.5), Value::new(-2.0)])[0].backward();
        assert!(m.parameters().iter().any(|p| p.grad() != 0.0));
        m.zero_grad();
        assert!(m.parameters().iter().all(|p| p.grad() == 0.0));
    }
}
