use std::{cell::RefCell, f32::consts::E};

pub struct Neuron {
    // TODO store neuron parameter as a Mx1 matrix
    weights: RefCell<Vec<f32>>,
    bias: f32,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + E.powf(-x))
}

impl Default for Neuron {
    fn default() -> Self {
        Self {
            weights: Default::default(),
            bias: Default::default(),
        }
    }
}

impl Neuron {
    pub fn predict(&self, inputs: &Vec<f32>) -> Option<f32> {
        let weights = self.weights.borrow();
        if weights.len() == inputs.len() {
            let mut output = 0.0;
            for (index, weight) in weights.iter().enumerate() {
                output += weight * inputs[index];
            }
            Some(output + self.bias)
        } else {
            None
        }
    }
}
