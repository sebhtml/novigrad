use crate::{
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
        let max_alpha = &device_stream.max_alpha;
        let l2_norm = &device_stream.l2_norm;
        ReduceL2::execute(
            &OperatorAttributes::None,
            &[&output],
            &[&l2_norm],
            device,
            device_stream,
        )?;
        let one = &device_stream.one;
        let alpha = &device_stream.alpha;
        device.div(&one, &l2_norm, &alpha, device_stream)?;
        device.min(&max_alpha, &alpha, &alpha, device_stream)?;
        device.scalar_mul(&alpha, output, device_stream)?;
        Ok(())
    }
}
