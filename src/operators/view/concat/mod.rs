use std::{ops::Deref, rc::Rc};

use crate::{Device, Error, NaryOperator, Operator, Tensor, TensorF32, Zero};

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

    pub fn concat(inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        let dst = outputs[0];
        for input_index in 0..inputs.len() {
            let src = inputs[input_index];
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

    pub fn unconcat(inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        let src = inputs[0];
        for input_index in 0..outputs.len() {
            let dst = outputs[input_index];
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

impl NaryOperator for Concat {
    fn forward(&self, inputs_n: &[&Tensor]) -> Result<Tensor, Error> {
        let rows = inputs_n[0].tensor().deref().borrow().rows();
        let cols = inputs_n[0].tensor().deref().borrow().cols();
        for input in inputs_n.iter() {
            debug_assert_eq!(input.tensor().deref().borrow().rows(), rows);
            debug_assert_eq!(input.tensor().deref().borrow().cols(), cols);
        }
        let cols = inputs_n.len() * cols;
        let len = rows * cols;
        let values = vec![0.0; len];
        let output = self
            .device
            .tensor(rows, cols, values, inputs_n, true, false);
        let inputs = inputs_n;
        let outputs = [&output];
        let inputs: Vec<TensorF32> = inputs
            .iter()
            .map(|t| t.tensor().deref().borrow().clone())
            .collect();
        output.push_instruction(
            Rc::new(Zero::default()),
            &[],
            &[&outputs[0].tensor().deref().borrow()],
            false,
        );
        output.push_instruction(
            Rc::new(Zero::default()),
            &[],
            &[&outputs[0].gradient().deref().borrow()],
            false,
        );
        output.push_instruction(
            Rc::new(self.clone()),
            &inputs.iter().collect::<Vec<_>>(),
            &[&outputs[0].tensor().deref().borrow()],
            false,
        );
        let inputs = [&output];
        let outputs = inputs_n;
        let outputs: Vec<TensorF32> = outputs
            .iter()
            .map(|t| t.gradient().deref().borrow().clone())
            .collect();
        output.push_backward_instruction(
            Rc::new(ConcatBackward::default()),
            &[&inputs[0].gradient().deref().borrow_mut()],
            &outputs.iter().collect::<Vec<_>>(),
            true,
        );
        Ok(output)
    }
}

impl Operator for Concat {
    fn name(&self) -> &str {
        "Concat"
    }

    fn forward(&self, inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        Self::concat(inputs, outputs)
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

    fn forward(&self, inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        Concat::unconcat(inputs, outputs)
    }
}
