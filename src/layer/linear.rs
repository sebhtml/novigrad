use rand::{distributions::Uniform, thread_rng, Rng};

use crate::{
    DeltaWorkingMemory, DifferentiableTensor, Error, Layer, Tensor, TRANSPOSE_LHS,
    TRANSPOSE_RESULT, TRANSPOSE_RHS,
};

pub struct Linear {
    weights: DifferentiableTensor,
    biases: DifferentiableTensor,
    tmp: Tensor,
}

impl Linear {
    pub fn new(input_rows: usize, rows: usize, cols: usize) -> Self {
        let mut rng = thread_rng();
        let mut weights = Vec::new();
        let right = (6.0 as f32).sqrt() / (cols as f32 + rows as f32).sqrt();
        let left = -right;
        // Xavier Initialization, or Glorot Initialization,
        let uniform = Uniform::new(left, right);
        weights.resize(rows * cols, 0.0);
        for index in 0..weights.len() {
            weights[index] = rng.sample(uniform);
        }
        let weights = Tensor::new(rows, cols, weights);
        let mut biases = Tensor::default();
        biases.reset(input_rows, rows, Default::default());
        Linear {
            weights: weights.into(),
            biases: biases.into(),
            tmp: Default::default(),
        }
    }
}

impl Layer for Linear {
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

    fn get_layer_delta(
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
    // TODO rename input_rows to bias_rows
    pub input_rows: usize,
    pub rows: usize,
    pub cols: usize,
}

impl Into<Linear> for &LinearConfig {
    fn into(self) -> Linear {
        Linear::new(self.input_rows, self.rows, self.cols)
    }
}
