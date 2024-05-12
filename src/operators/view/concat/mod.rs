use std::{ops::Deref, rc::Rc};

use crate::{Device, Error, NaryOperator, Operator, Tensor, TensorF32};

#[cfg(test)]
mod tests;

#[derive(Clone)]
pub struct Concat {
    device: Device,
}

impl Concat {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }
}

impl NaryOperator for Concat {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, Error> {
        let rows = inputs[0].tensor().deref().borrow().rows();
        let cols = inputs[0].tensor().deref().borrow().cols();
        for input in inputs.iter() {
            debug_assert_eq!(input.tensor().deref().borrow().rows(), rows);
            debug_assert_eq!(input.tensor().deref().borrow().cols(), cols);
        }
        let cols = inputs.len() * cols;
        let len = rows * cols;
        let values = vec![0.0; len];
        let output = self.device.tensor(rows, cols, values, true, false);
        let outputs = &[&output];
        output.push_forward_instruction(Rc::new(self.clone()), inputs, outputs);
        output.push_backward_instruction(Rc::new(ConcatBackward::default()), outputs, inputs);
        Ok(output)
    }
}

impl Operator for Concat {
    fn name(&self) -> &str {
        "Concat"
    }

    fn forward(&self, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        let dst = outputs[0].tensor().deref().borrow();
        for input_index in 0..inputs.len() {
            let src = inputs[input_index].tensor().deref().borrow();
            let src_col = 0;
            let input_rows = src.rows();
            let input_cols = src.cols();
            for src_row in 0..input_rows {
                let dst_row = src_row;
                let dst_col = input_index * input_cols;
                TensorF32::copy_slice(&src, src_row, src_col, &dst, dst_row, dst_col)?;
            }
        }
        Ok(())
    }
}

pub struct ConcatBackward {}

impl Default for ConcatBackward {
    fn default() -> Self {
        Self {}
    }
}

impl Operator for ConcatBackward {
    fn name(&self) -> &str {
        "ConcatBackward"
    }

    fn forward(&self, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        let src: &TensorF32 = &inputs[0].gradient().deref().borrow_mut();
        for input_index in 0..outputs.len() {
            let dst: &mut TensorF32 = &mut outputs[input_index].gradient().deref().borrow_mut();
            let dst_col = 0;
            let input_rows = dst.rows();
            let input_cols = dst.cols();
            for dst_row in 0..input_rows {
                let src_row = dst_row;
                let src_col = input_index * input_cols;
                TensorF32::copy_slice(src, src_row, src_col, dst, dst_row, dst_col)?;
            }
        }
        Ok(())
    }
}
