use crate::{DeviceInterface, Error, Tensor};

pub struct Bernoulli {}

impl Bernoulli {
    pub fn execute(probability: f32, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        let device = input.device();
        device.bernoulli(probability, input, output)
    }
}
