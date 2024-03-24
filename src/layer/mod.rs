mod linear;
use std::{cell::RefCell, rc::Rc};

pub use linear::*;

use crate::{ActivationFunction, Error, Tensor};

pub trait Layer {
    fn weights(&self) -> Rc<RefCell<Tensor>>;
    fn activation(&self) -> Rc<dyn ActivationFunction>;
    fn forward(&self, input: &Tensor, w_t: &mut Tensor, result: &mut Tensor) -> Result<(), Error>;
}
