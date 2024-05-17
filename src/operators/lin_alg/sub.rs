use crate::{Error, TensorF32};

/// https://onnx.ai/onnx/operators/onnx__Add.html
#[derive(Clone)]
pub struct Sub {}

impl Sub {
    pub fn execute(inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        let input_0 = inputs[0];
        let input_1 = inputs[1];
        let output = outputs[0];
        TensorF32::copy(input_0, output)?;
        TensorF32::sub(input_1, output)
    }
}
