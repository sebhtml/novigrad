use std::{cell::RefCell, f32::consts::E};

pub struct Neuron {
    // TODO store neuron parameter as a Mx1 matrix
    pub weights: RefCell<Vec<f32>>,
    pub bias: RefCell<f32>,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + E.powf(-x))
}

impl Default for Neuron {
    fn default() -> Self {
        Self {
            weights: RefCell::new(vec![0.5]),
            bias: RefCell::new(0.0),
        }
    }
}

impl Neuron {
    pub fn predict(&self, inputs: &Vec<f32>) -> f32 {
        let weights = self.weights.borrow();
        let bias: f32 = *self.bias.borrow();
        if weights.len() == inputs.len() {
            let mut output = 0.0;
            for (index, weight) in weights.iter().enumerate() {
                output += weight * inputs[index];
            }
            sigmoid(output - bias)
        } else {
            Default::default()
        }
    }
}
