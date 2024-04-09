use rand::{distributions::Uniform, thread_rng, Rng};

use crate::{
    DeltaWorkingMemory, DifferentiableModuleTrait, DifferentiableTensor, Error, Tensor,
    TRANSPOSE_LHS, TRANSPOSE_RESULT, TRANSPOSE_RHS,
};

pub struct Linear {
    weights: DifferentiableTensor,
    biases: DifferentiableTensor,
    tmp: Tensor,
}

impl Linear {
    pub fn new(weights_rows: usize, weights_cols: usize, bias_rows: usize) -> Self {
        // Xavier Initialization, or Glorot Initialization,
        let mut rng = thread_rng();
        let right = (6.0 as f32).sqrt() / (weights_cols as f32 + weights_rows as f32).sqrt();
        let left = -right;
        let uniform = Uniform::new(left, right);

        let mut weights = Vec::new();
        weights.resize(weights_rows * weights_cols, 0.0);
        for index in 0..weights.len() {
            weights[index] = rng.sample(uniform);
        }
        let weights = Tensor::new(weights_rows, weights_cols, weights);

        let mut biases = Tensor::default();
        biases.reset(bias_rows, weights_rows, Default::default());

        Linear {
            weights: weights.into(),
            biases: biases.into(),
            tmp: Default::default(),
        }
    }
}

impl DifferentiableModuleTrait for Linear {
    fn commit_change(&mut self, learning_rate: f32) -> Result<(), Error> {
        self.weights.commit_change(learning_rate);
        self.biases.commit_change(learning_rate);
        Ok(())
    }

    fn forward(&mut self, input: &Tensor, output: &mut Tensor) -> Result<(), Error> {
        // Use the same convention that is used in tensorflow:
        // y = x @ W^T+b
        // Weights is on the right.
        // X is not transposed.
        // W is transposed.
        let weights = &self.weights.tensor;
        let tmp = &mut self.tmp;
        let op_result = Tensor::matmul(input, weights, tmp, TRANSPOSE_RHS);
        match op_result {
            Ok(_) => (),
            Err(_) => {
                let mut w_t = Tensor::default();
                weights.transpose(&mut w_t);
                println!("Incompatible shapes in matrix multiplication");
                println!("Between X {:?} and W^T {:?}", input.shape(), w_t.shape(),);
                debug_assert!(false);
            }
        }

        let biases = &self.biases.tensor;
        let op_result = tmp.add(biases, output);
        match op_result {
            Ok(_) => (),
            Err(_) => {
                println!("Incompatible shapes in matrix multiplication");
                println!("Between A {:?} and B {:?}", tmp.shape(), biases.shape(),);
                debug_assert!(false);
            }
        }

        op_result.expect("Ok");
        Ok(())
    }

    fn backward(&self, layer_output_delta: &Tensor, previous_layer_output_delta: &mut Tensor) {
        let weights = &self.weights.tensor;

        let op_result = Tensor::matmul(
            weights,
            layer_output_delta,
            previous_layer_output_delta,
            TRANSPOSE_LHS | TRANSPOSE_RHS | TRANSPOSE_RESULT,
        );

        op_result.expect("Ok");
    }

    fn get_layer_output_delta(
        &self,
        _working_memory: &mut DeltaWorkingMemory,
        _layer_input: &Tensor,
        _layer_output: &Tensor,
        back_propagated_delta: &Tensor,
        _is_last_layer: bool,
        layer_delta: &mut Tensor,
    ) {
        layer_delta.assign(back_propagated_delta)
    }

    fn compute_gradient(&mut self, layer_input: &Tensor, layer_output_delta: &Tensor) {
        let op_result = Tensor::matmul(
            layer_input,
            layer_output_delta,
            &mut self.weights.gradient,
            TRANSPOSE_LHS | TRANSPOSE_RESULT,
        );
        op_result.expect("Ok");
        self.weights.has_gradient = true;

        self.biases.gradient.assign(layer_output_delta);
        self.biases.has_gradient = true;
    }
}

pub struct LinearConfig {
    pub weights_rows: usize,
    pub weights_cols: usize,
    pub bias_rows: usize,
}

impl Into<Linear> for &LinearConfig {
    fn into(self) -> Linear {
        Linear::new(self.weights_rows, self.weights_cols, self.bias_rows)
    }
}
