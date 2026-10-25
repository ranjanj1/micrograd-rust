//! A Rust port of micrograd: a tiny scalar autograd engine.

#![forbid(unsafe_code)]

pub mod engine;
pub mod nn;
pub mod train;

pub use engine::Value;
pub use nn::{Layer, MLP, Module, Neuron};
