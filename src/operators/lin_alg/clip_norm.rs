use crate::{
    stream::DeviceStream,
    tensor::{Error, Tensor},
};

pub struct ClipNorm {}

impl ClipNorm {
    pub fn execute(
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        if input.name() != output.name() {
            Tensor::copy(input, output, device_stream)?;
        }
        output.clip_norm()?;
        Ok(())
    }
}
