use rand::{distributions::Uniform, thread_rng, Rng};

use crate::{Error, Tensor};

pub struct Bernoulli {}

impl Bernoulli {
    pub fn execute(probability: f32, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        let len = input.len();
        let mut values = vec![1.0; len];
        let mut rng = thread_rng();
        let uniform = Uniform::new(0.0, 1.0);

        for i in 0..len {
            let random_number = if rng.sample(uniform) <= probability {
                1.0
            } else {
                0.0
            };
            values[i] = random_number;
        }
        output.set_values(values)
    }
}
