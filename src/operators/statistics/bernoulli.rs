use rand::{thread_rng, Rng};
use rand_distr::Uniform;

use crate::{
    stream::DeviceStream,
    tensor::{Error, Tensor},
    Device, ExecutableOperator, OperatorAttributes,
};

pub struct Bernoulli {}

impl ExecutableOperator for Bernoulli {
    fn execute(
        attributes: &OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        _device: &Device,
        _device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        let n = input.len();
        let probability = match attributes {
            OperatorAttributes::F32(probability) => *probability,
            _ => panic!("No probability was provided"),
        };
        let trials = bernoulli(n, probability);
        output.set_values(trials)?;
        Ok(())
    }
}

fn bernoulli(n: usize, probability: f32) -> Vec<f32> {
    let mut rng = thread_rng();
    let uniform = Uniform::new(0.0, 1.0);

    (0..n)
        .map(|_| {
            let random_number = if rng.sample(uniform) <= probability {
                1.0
            } else {
                0.0
            };
            random_number
        })
        .collect::<Vec<_>>()
}
