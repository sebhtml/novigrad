use crate::{
    error,
    stream::DeviceStream,
    tensor::{Error, ErrorEnum, Tensor},
    Device, DeviceTrait, ExecutableOperator, OperatorAttributes,
};

pub struct Sub {}

impl ExecutableOperator for Sub {
    fn execute(
        _attributes: &OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device: &Device,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let input_0 = inputs[0];
        let input_1 = inputs[1];
        let output = outputs[0];
        if input_0.len() != input_1.len() {
            println!("Incompatible sizes");
            println!("x {}", input_0);
            println!("y {}", input_1);
            return Err(error!(ErrorEnum::IncompatibleTensorShapes));
        }
        if input_0.len() != output.len() {
            println!("Incompatible sizes");
            println!("x {}", input_0);
            println!("y {}", output);
            return Err(error!(ErrorEnum::IncompatibleTensorShapes));
        }
        device.copy_to(input_0, output, device_stream)?;

        let alpha = -1.0;

        let n = input_1.len() as i32;
        let incx = 1;
        let incy = 1;
        device.axpy(n, alpha, input_1, incx, output, incy, device_stream)
    }
}
