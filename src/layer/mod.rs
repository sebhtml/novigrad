mod linear;
use std::{cell::RefCell, rc::Rc};

pub use linear::*;

use crate::{ActivationFunction, Tensor};

pub trait Layer {
    fn weights(&self) -> Rc<RefCell<Tensor>>;
    fn activation(&self) -> Rc<dyn ActivationFunction>;
}
