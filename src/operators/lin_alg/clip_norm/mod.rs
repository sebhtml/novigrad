use crate::{
    new_tensor,
    reduce_l2::ReduceL2,
    stream::DeviceStream,
    tensor::{Error, Tensor},
    Device, DeviceTrait, ExecutableOperator, OperatorAttributes,
};

#[cfg(test)]
mod tests;

pub struct ClipNorm {}

impl ExecutableOperator for ClipNorm {
    fn execute(
        _attributes: &OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device: &Device,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        if input.name() != output.name() {
            device.copy_to(input, output, device_stream)?;
        }
        let max_alpha = new_tensor!(device, 1, 1, vec![1.0],)?;
        let l2_norm = new_tensor!(device, 1, 1, vec![0.0],)?;
        ReduceL2::execute(
            &OperatorAttributes::None,
            &[&output],
            &[&l2_norm],
            device,
            device_stream,
        )?;
        let one = new_tensor!(device, 1, 1, vec![1.0],)?;
        let alpha = new_tensor!(device, 1, 1, vec![0.0],)?;
        device.div(&one, &l2_norm, &alpha, device_stream)?;
        device.min(&max_alpha, &alpha, &alpha, device_stream)?;
        device.scalar_mul(&alpha, output, device_stream)?;
        Ok(())
    }
}
