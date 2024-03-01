pub struct Network {}

impl Default for Network {
    fn default() -> Self {
        Self {}
    }
}

impl Network {
    pub fn train(&self, inputs: &Vec<Vec<f32>>, outputs: &Vec<Vec<f32>>) {}

    pub fn back_propagation(&self, input: &Vec<f32>, output: &Vec<f32>, target: &Vec<f32>) {}

    pub fn predict_many(&self, inputs: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        inputs.iter().map(|x| self.predict(x)).collect()
    }

    pub fn predict(&self, input: &Vec<f32>) -> Vec<f32> {
        vec![]
    }
}
