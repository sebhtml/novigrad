use std::ops::Index;

use crate::Neuron;

pub struct Network {
    neuron: Neuron,
}

impl Default for Network {
    fn default() -> Self {
        Self {
            neuron: Default::default(),
        }
    }
}

impl Network {
    pub fn train(&self, inputs: &Vec<Vec<f32>>, outputs: &Vec<Vec<f32>>) {
        /*
        If we have N examples with <WIDTH> inputs, that's a matrix with <N> rows and <WIDTH> columns
         */
        for (index, _) in inputs.iter().enumerate() {
            let input = &inputs[index];
            let output = &outputs[index];
            let predicted = self.predict(input);
            self.back_propagation(input, &predicted, output, );
        }
    }

    pub fn back_propagation(&self, input: &Vec<f32>, output: &Vec<f32>, target: &Vec<f32>) {
        // see https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
        // For neuron 0
        let target_o1 = target[0];
        let output_o1 = {
            let value = output[0];
            if value == 1.0 {
                0.99
            } else if value == 0.0 {
                0.01
            } else {
                value
            }
        };
        let error_o1 = 0.5 * (target_o1 - output_o1);
        println!("target_o1 {}  output_o1 {}  error_o1 {}", target_o1, output_o1, error_o1);
        let error_total = error_o1;
        let d_error_total_out_o1 = -(target_o1 - output_o1);
        let d_out_o1_net_o1 = output_o1 * (1.0 - output_o1);
        let out_h1 = input[0]; // TODO store values in a forward pass.
        let d_net_o1_w1 = out_h1;
        println!("d_error_total_out_o1 {} d_out_o1_net_o1 {}  d_net_o1_w1 {} ", d_error_total_out_o1, d_out_o1_net_o1, d_net_o1_w1);
        let d_error_total_w1 = d_error_total_out_o1 * d_out_o1_net_o1 * d_net_o1_w1;
        let learning_rate = 0.5;
        let old_weight = self.neuron.weights.borrow_mut()[0];
        self.neuron.weights.borrow_mut()[0] -= learning_rate * d_error_total_w1;
        let new_weight = self.neuron.weights.borrow_mut()[0];
        println!("Updated weight {:?} -> {:?} ", old_weight, new_weight);
    }

    pub fn predict_many(&self, inputs: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        inputs.iter().map(|x| self.predict(x)).collect()
    }

    pub fn predict(&self, input: &Vec<f32>) -> Vec<f32> {
        vec![self.neuron.predict(input)]
    }
}
