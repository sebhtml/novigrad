use std::ops::Deref;

use crate::{
    gradient_instruction, inference_instruction, Device, Error, GenericTensor, NaryOperator,
    OpCode, Tensor,
};

#[cfg(test)]
mod tests;

pub struct Concat {
    device: Device,
}

impl Concat {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }

    pub fn execute(inputs: &[&GenericTensor], outputs: &[&GenericTensor]) -> Result<(), Error> {
        let dst = outputs[0];
        for input_index in 0..inputs.len() {
            let src = inputs[input_index];
            let src_col = 0;
            let input_rows = src.rows();
            let input_cols = src.cols();
            for src_row in 0..input_rows {
                let dst_row = src_row;
                let dst_col = input_index * input_cols;
                GenericTensor::copy_slice(
                    src.cols(),
                    &src,
                    src_row,
                    src_col,
                    &dst,
                    dst_row,
                    dst_col,
                )?;
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
        let inputs: Vec<GenericTensor> = inputs
            .iter()
            .map(|t| t.tensor().deref().borrow().clone())
            .collect();
        output.push_instruction(inference_instruction!(
            OpCode::ScalarMul(0.0),
            &[&outputs[0].tensor().deref().borrow()],
            &[&outputs[0].tensor().deref().borrow()],
        ));
        output.push_instruction(inference_instruction!(
            OpCode::ScalarMul(0.0),
            &[&outputs[0].gradient().deref().borrow()],
            &[&outputs[0].gradient().deref().borrow()],
        ));
        output.push_instruction(inference_instruction!(
            OpCode::Concat,
            &inputs.iter().collect::<Vec<_>>(),
            &[&outputs[0].tensor().deref().borrow()],
        ));
        let inputs = [&output];
        let outputs = inputs_n;
        let outputs: Vec<GenericTensor> = outputs
            .iter()
            .map(|t| t.gradient().deref().borrow().clone())
            .collect();
        output.push_instruction(gradient_instruction!(
            OpCode::Unconcat,
            &[&inputs[0].gradient().deref().borrow_mut()],
            &outputs.iter().collect::<Vec<_>>(),
        ));
        Ok(output)
    }
}

pub struct Unconcat {}

impl Unconcat {
    pub fn execute(inputs: &[&GenericTensor], outputs: &[&GenericTensor]) -> Result<(), Error> {
        let src = inputs[0];
        for output_index in 0..outputs.len() {
            let dst = outputs[output_index];
            let dst_col = 0;
            let input_rows = dst.rows();
            let input_cols = dst.cols();
            for dst_row in 0..input_rows {
                let src_row = dst_row;
                let src_col = output_index * input_cols;
                GenericTensor::copy_slice(
                    dst.cols(),
                    src,
                    src_row,
                    src_col,
                    dst,
                    dst_row,
                    dst_col,
                )?;
            }
        }
        Ok(())
    }
}
