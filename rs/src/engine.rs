//! Scalar autograd: a `Value` records the operation that produced it, and
//! `backward` walks that graph in reverse applying the chain rule.
//!
//! Every operation is built and ordered exactly like `micrograd/engine.py` (with
//! children kept in `(self, other)` order), so results match it bit for bit.

use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

/// A scalar in the computation graph. Cloning is cheap and shares the same node.
#[derive(Clone)]
pub struct Value(Rc<RefCell<Node>>);

struct Node {
    data: f64,
    grad: f64,
    op: Op,
}

/// The operation that produced a node, holding its children in `(self, other)` order.
enum Op {
    Leaf,
    Add(Value, Value),
    Mul(Value, Value),
    Pow(Value, f64),
    Relu(Value),
}

impl Value {
    pub fn new(data: f64) -> Self {
        Self::from_op(data, Op::Leaf)
    }

    fn from_op(data: f64, op: Op) -> Self {
        Value(Rc::new(RefCell::new(Node {
            data,
            grad: 0.0,
            op,
        })))
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    pub fn set_data(&self, data: f64) {
        self.0.borrow_mut().data = data;
    }

    pub fn set_grad(&self, grad: f64) {
        self.0.borrow_mut().grad = grad;
    }

    fn add_grad(&self, delta: f64) {
        self.0.borrow_mut().grad += delta;
    }

    pub fn pow(&self, n: f64) -> Value {
        Value::from_op(self.data().powf(n), Op::Pow(self.clone(), n))
    }

    pub fn relu(&self) -> Value {
        let data = self.data();
        Value::from_op(if data < 0.0 { 0.0 } else { data }, Op::Relu(self.clone()))
    }

    /// Fill in `grad` for every node this value depends on.
    pub fn backward(&self) {
        let topo = self.topo_order();
        self.set_grad(1.0);
        for v in topo.iter().rev() {
            v.backward_step();
        }
    }

    /// Post-order DFS over children in stored order, the same order as Python's
    /// recursive `build_topo`. Iterative so deep graphs can't overflow the stack.
    fn topo_order(&self) -> Vec<Value> {
        let mut topo = Vec::new();
        let mut visited = HashSet::from([self.ptr()]);
        let mut stack = vec![(self.clone(), 0)];
        while let Some((node, i)) = stack.last_mut() {
            match node.child(*i) {
                Some(child) => {
                    *i += 1;
                    if visited.insert(child.ptr()) {
                        stack.push((child, 0));
                    }
                }
                None => {
                    if let Some((node, _)) = stack.pop() {
                        topo.push(node);
                    }
                }
            }
        }
        topo
    }

    fn child(&self, i: usize) -> Option<Value> {
        match (&self.0.borrow().op, i) {
            (Op::Add(a, _) | Op::Mul(a, _) | Op::Pow(a, _) | Op::Relu(a), 0) => Some(a.clone()),
            (Op::Add(_, b) | Op::Mul(_, b), 1) => Some(b.clone()),
            _ => None,
        }
    }

    /// Node identity for the visited set; the pointer is only compared, never read.
    fn ptr(&self) -> *const RefCell<Node> {
        Rc::as_ptr(&self.0)
    }

    /// Push this node's grad to its children (Python's `_backward`).
    fn backward_step(&self) {
        let node = self.0.borrow();
        let g = node.grad;
        match &node.op {
            Op::Leaf => {}
            Op::Add(a, b) => {
                a.add_grad(g);
                b.add_grad(g);
            }
            Op::Mul(a, b) => {
                let (a_data, b_data) = (a.data(), b.data());
                a.add_grad(b_data * g);
                b.add_grad(a_data * g);
            }
            Op::Pow(a, n) => a.add_grad((n * a.data().powf(n - 1.0)) * g),
            Op::Relu(a) => a.add_grad(if node.data > 0.0 { 1.0 } else { 0.0 } * g),
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value(data={}, grad={})", self.data(), self.grad())
    }
}

// The basic ops. Constants become leaf nodes, and a constant on the left still
// goes second (Python's `__radd__`/`__rmul__` call `self + other`).

fn add_vv(a: &Value, b: &Value) -> Value {
    Value::from_op(a.data() + b.data(), Op::Add(a.clone(), b.clone()))
}

fn add_vf(a: &Value, b: f64) -> Value {
    add_vv(a, &Value::new(b))
}

fn add_fv(a: f64, b: &Value) -> Value {
    add_vv(b, &Value::new(a))
}

fn mul_vv(a: &Value, b: &Value) -> Value {
    Value::from_op(a.data() * b.data(), Op::Mul(a.clone(), b.clone()))
}

fn mul_vf(a: &Value, b: f64) -> Value {
    mul_vv(a, &Value::new(b))
}

fn mul_fv(a: f64, b: &Value) -> Value {
    mul_vv(b, &Value::new(a))
}

// Derived ops, composed exactly as in Python so rounding matches.

fn neg(a: &Value) -> Value {
    mul_vf(a, -1.0)
}

fn sub_vv(a: &Value, b: &Value) -> Value {
    add_vv(a, &neg(b))
}

fn sub_vf(a: &Value, b: f64) -> Value {
    add_vf(a, -b)
}

fn sub_fv(a: f64, b: &Value) -> Value {
    add_fv(a, &neg(b))
}

fn div_vv(a: &Value, b: &Value) -> Value {
    mul_vv(a, &b.pow(-1.0))
}

fn div_vf(a: &Value, b: f64) -> Value {
    mul_vf(a, b.powf(-1.0))
}

fn div_fv(a: f64, b: &Value) -> Value {
    mul_fv(a, &b.pow(-1.0))
}

/// Implement an operator for every owned/borrowed mix of `Value` and `f64`.
macro_rules! impl_op {
    ($Trait:ident, $method:ident, $vv:ident, $vf:ident, $fv:ident) => {
        impl $Trait<&Value> for &Value {
            type Output = Value;
            fn $method(self, rhs: &Value) -> Value {
                $vv(self, rhs)
            }
        }
        impl $Trait<Value> for &Value {
            type Output = Value;
            fn $method(self, rhs: Value) -> Value {
                $vv(self, &rhs)
            }
        }
        impl $Trait<&Value> for Value {
            type Output = Value;
            fn $method(self, rhs: &Value) -> Value {
                $vv(&self, rhs)
            }
        }
        impl $Trait<Value> for Value {
            type Output = Value;
            fn $method(self, rhs: Value) -> Value {
                $vv(&self, &rhs)
            }
        }
        impl $Trait<f64> for &Value {
            type Output = Value;
            fn $method(self, rhs: f64) -> Value {
                $vf(self, rhs)
            }
        }
        impl $Trait<f64> for Value {
            type Output = Value;
            fn $method(self, rhs: f64) -> Value {
                $vf(&self, rhs)
            }
        }
        impl $Trait<&Value> for f64 {
            type Output = Value;
            fn $method(self, rhs: &Value) -> Value {
                $fv(self, rhs)
            }
        }
        impl $Trait<Value> for f64 {
            type Output = Value;
            fn $method(self, rhs: Value) -> Value {
                $fv(self, &rhs)
            }
        }
    };
}

impl_op!(Add, add, add_vv, add_vf, add_fv);
impl_op!(Mul, mul, mul_vv, mul_vf, mul_fv);
impl_op!(Sub, sub, sub_vv, sub_vf, sub_fv);
impl_op!(Div, div, div_vv, div_vf, div_fv);

impl Neg for &Value {
    type Output = Value;
    fn neg(self) -> Value {
        neg(self)
    }
}

impl Neg for Value {
    type Output = Value;
    fn neg(self) -> Value {
        neg(&self)
    }
}

/// Python's built-in `sum`, which starts from 0: `((v0 + 0) + v1) + ...`.
/// The extra constant node keeps the graph identical to Python's.
impl Sum for Value {
    fn sum<I: Iterator<Item = Value>>(mut iter: I) -> Value {
        match iter.next() {
            None => Value::new(0.0),
            Some(first) => iter.fold(first + 0.0, |acc, v| acc + v),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Port of test_sanity_check in test/test_engine.py. Expected values from torch;
    // micrograd matches them exactly, so we compare with ==.
    #[test]
    fn sanity_check() {
        let x = Value::new(-4.0);
        let z = 2.0 * &x + 2.0 + &x;
        let q = z.relu() + &z * &x;
        let h = (&z * &z).relu();
        let y = &h + &q + &q * &x;
        y.backward();
        assert_eq!(y.data(), -20.0);
        assert_eq!(x.grad(), 46.0);
    }

    // Port of test_more_ops; `c += expr` in Python means `c = c + expr`.
    #[test]
    fn more_ops() {
        let a = Value::new(-4.0);
        let b = Value::new(2.0);
        let mut c = &a + &b;
        let mut d = &a * &b + b.pow(3.0);
        c = &c + (&c + 1.0);
        c = &c + (1.0 + &c + (-&a));
        d = &d + (&d * 2.0 + (&b + &a).relu());
        d = &d + (3.0 * &d + (&b - &a).relu());
        let e = &c - &d;
        let f = e.pow(2.0);
        let mut g = &f / 2.0;
        g = &g + 10.0 / &f;
        g.backward();
        assert_eq!(g.data(), 24.70408163265306);
        assert_eq!(a.grad(), 138.83381924198252);
        assert_eq!(b.grad(), 645.5772594752186);
    }

    #[test]
    fn add_and_mul() {
        let a = Value::new(2.0);
        let b = Value::new(-3.0);
        let c = &a * &b + &a;
        c.backward();
        assert_eq!(c.data(), -4.0);
        assert_eq!(a.grad(), -2.0); // b + 1
        assert_eq!(b.grad(), 2.0); // a
    }

    #[test]
    fn square_uses_same_child_twice() {
        let x = Value::new(3.0);
        let y = &x * &x;
        y.backward();
        assert_eq!(x.grad(), 6.0);
    }

    #[test]
    fn reused_node_accumulates_grad() {
        let x = Value::new(1.5);
        let y = &x * 2.0 + &x * 3.0;
        y.backward();
        assert_eq!(x.grad(), 5.0);
    }

    #[test]
    fn pow() {
        let x = Value::new(2.0);
        let y = x.pow(3.0);
        y.backward();
        assert_eq!(y.data(), 8.0);
        assert_eq!(x.grad(), 12.0);

        let x = Value::new(4.0);
        let y = x.pow(-1.0);
        y.backward();
        assert_eq!(y.data(), 0.25);
        assert_eq!(x.grad(), -0.0625);
    }

    #[test]
    fn relu() {
        for (input, out, grad) in [(2.5, 2.5, 1.0), (-2.5, 0.0, 0.0), (0.0, 0.0, 0.0)] {
            let x = Value::new(input);
            let y = x.relu();
            y.backward();
            assert_eq!(y.data(), out, "relu({input})");
            assert_eq!(x.grad(), grad, "relu'({input})");
        }
    }

    #[test]
    fn sub_neg_div() {
        let a = Value::new(6.0);
        let b = Value::new(3.0);
        let y = &a - &b + -&a + &a / &b;
        y.backward();
        assert_eq!(y.data(), -1.0);
        // Exact values from the Python reference: a's grad is 1 - 1 + 1/3 summed in
        // backward order, which rounds differently from evaluating 1.0 / 3.0 directly.
        assert_eq!(a.grad(), 0.33333333333333326);
        assert_eq!(b.grad(), -1.6666666666666665);
    }

    #[test]
    fn constant_on_either_side() {
        let x = Value::new(4.0);
        let y = 1.0 - &x + (&x - 1.0) + 8.0 / &x + &x / 2.0;
        y.backward();
        assert_eq!(y.data(), -3.0 + 3.0 + 2.0 + 2.0);
        assert_eq!(x.grad(), -1.0 + 1.0 - 8.0 / 16.0 + 0.5);
    }

    #[test]
    fn backward_on_leaf_sets_grad_to_one() {
        let x = Value::new(7.0);
        x.backward();
        assert_eq!(x.grad(), 1.0);
    }

    #[test]
    fn sum_starts_from_zero_like_python() {
        let a = Value::new(1.0);
        let b = Value::new(2.0);
        let s: Value = [a.clone(), b.clone()].into_iter().sum();
        s.backward();
        assert_eq!(s.data(), 3.0);
        assert_eq!((a.grad(), b.grad()), (1.0, 1.0));
        // a, the constant 0, a + 0, b, s
        assert_eq!(s.topo_order().len(), 5);

        let empty: Value = std::iter::empty().sum();
        assert_eq!(empty.data(), 0.0);
    }

    // Python (with children pinned to insertion order) visits y = (a * b) + a as
    // [a, b, a*b, y]; the gradient summation order depends on this.
    #[test]
    fn topo_order_matches_python() {
        let a = Value::new(2.0);
        let b = Value::new(5.0);
        let ab = &a * &b;
        let y = &ab + &a;
        let order: Vec<_> = y.topo_order().iter().map(Value::ptr).collect();
        assert_eq!(order, [a.ptr(), b.ptr(), ab.ptr(), y.ptr()]);
    }
}
