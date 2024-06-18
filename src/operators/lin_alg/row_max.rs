use crate::{
    tensor::{Error, Tensor},
    ExecutableOperator,
};

// TODO rename to https://onnx.ai/onnx/operators/onnx__ArgMax.html
pub struct RowMax {}

impl ExecutableOperator for RowMax {
    fn execute(
        _attributes: &crate::OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        _device: &crate::Device,
        _device_stream: &crate::stream::DeviceStream,
    ) -> Result<(), Error> {
        let _input = inputs[0];
        let _output = outputs[0];
        //device.row_max(input, output)
        Ok(())
    }
}
