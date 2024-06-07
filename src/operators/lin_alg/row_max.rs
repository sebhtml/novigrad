use crate::tensor::{Error, Tensor};

// TODO rename to https://onnx.ai/onnx/operators/onnx__ArgMax.html
pub struct RowMax {}

impl RowMax {
    pub fn execute(inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        let input = inputs[0];
        let _output = outputs[0];
        let _device = input.device();
        //device.row_max(input, output)
        Ok(())
    }
}
