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
    pub fn train(inputs: Vec<Vec<f32>>, output: Vec<Vec<f32>>) {
        /*
        If we have N examples with <WIDTH> inputs, that's a matrix with <N> rows and <WIDTH> columns
         */
    }

    pub fn predict_many(&self, inputs: &Vec<Vec<f32>>) -> Vec<Vec<Option<f32>>> {
        inputs.iter().map(|x| self.predict(x)).collect()
    }

    pub fn predict(&self, inputs: &Vec<f32>) -> Vec<Option<f32>> {
        vec![self.neuron.predict(inputs)]
    }
}
