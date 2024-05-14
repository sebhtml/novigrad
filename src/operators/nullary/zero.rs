use std::ops::Deref;

use crate::{Error, Operator, Tensor, TensorF32};

pub struct Zero {}

impl Default for Zero {
    fn default() -> Self {
        Self {}
    }
}

impl Operator for Zero {
    fn name(&self) -> &str {
        "Zero"
    }

    fn forward(&self, _inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        let t_outputs: Vec<TensorF32> = outputs
            .iter()
            .map(|t| t.tensor().deref().borrow().clone())
            .collect();
        let g_outputs: Vec<TensorF32> = outputs
            .iter()
            .map(|t| t.gradient().deref().borrow().clone())
            .collect();
        let outputs = t_outputs
            .into_iter()
            .chain(g_outputs.into_iter())
            .collect::<Vec<TensorF32>>();
        self.forward_f32(&[], &outputs.iter().collect::<Vec<_>>())
    }

    fn forward_f32(&self, _inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        for output in outputs {
            output.zero()?;
        }
        Ok(())
    }
}
