use crate::{
    new_tensor,
    reduce_l2::ReduceL2,
    stream::DeviceStream,
    tensor::{Error, Tensor},
    DeviceTrait, ExecutableOperator, OperatorAttributes,
};

#[cfg(test)]
mod tests;

pub struct ClipNorm {}

impl ExecutableOperator for ClipNorm {
    fn execute(
        _attributes: &OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        let device = input.device();
        if input.name() != output.name() {
            device.copy_to(input, output, device_stream)?;
        }
        let norm_max = 1.0;
        let l2_norm = new_tensor!(device, 1, 1, vec![0.0],)?;
        ReduceL2::execute(
            &OperatorAttributes::None,
            &[&output],
            &[&l2_norm],
            device_stream,
        )?;
        // TODO don't use get_values here.
        let l2_norm = l2_norm.get_values()?[0];
        // Can not normalize a vector with no direction.
        if l2_norm == 0.0 {
            return Ok(());
        }
        if l2_norm <= norm_max {
            return Ok(());
        }
        let alpha = 1.0 / l2_norm;
        let alpha = new_tensor!(device, 1, 1, vec![alpha],)?;
        device.scalar_mul(&alpha, output, device_stream)?;
        Ok(())
    }
}
