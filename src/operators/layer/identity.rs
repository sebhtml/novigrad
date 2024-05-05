use std::{ops::Deref, rc::Rc};

use crate::{Error, OperatorTrait, Tensor, TensorF32};

#[derive(Clone)]
pub struct Identity {}

impl Default for Identity {
    fn default() -> Self {
        Self {}
    }
}

impl OperatorTrait for Identity {
    fn name(&self) -> &str {
        "Identity"
    }

    fn forward(&self, device: &crate::Device, inputs: &[Tensor]) -> Result<Tensor, Error> {
        let input: &TensorF32 = &inputs[0].tensor().deref().borrow();
        let rows = input.rows();
        let cols = input.cols();
        let len = rows * cols;
        let output = device.tensor(
            Rc::new(self.clone()),
            inputs,
            rows,
            cols,
            vec![0.0; len],
            false,
        );
        {
            let output: &mut TensorF32 = &mut output.tensor().deref().borrow_mut();
            TensorF32::copy(input, output)?;
        }
        Ok(output)
    }

    fn backward(
        &self,
        device: &crate::Device,
        inputs: &[Tensor],
        output: &Tensor,
    ) -> Result<(), Error> {
        let output_gradient: &TensorF32 = &output.gradient().deref().borrow();
        let backward_gradient: &mut TensorF32 = &mut inputs[0].gradient().deref().borrow_mut();
        TensorF32::copy(output_gradient, backward_gradient)?;
        Ok(())
    }
}
