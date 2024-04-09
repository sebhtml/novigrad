use std::mem::swap;

use rand::{distributions::Uniform, thread_rng, Rng};

use crate::{
    DeltaWorkingMemory, Error, Layer, Tensor, TRANSPOSE_LHS, TRANSPOSE_RESULT,
    TRANSPOSE_RHS,
};

pub struct Linear {
    weights: Tensor,
    biases: Tensor,
    weights_delta: Tensor,
    biases_delta: Tensor,
    has_pending_change: bool,
    tmp: Tensor,
    addition: Tensor,
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
            weights,
            biases,
            weights_delta: Default::default(),
            biases_delta: Default::default(),
            has_pending_change: false,
            tmp: Default::default(),
            addition: Default::default(),
        }
    }
}

impl Layer for Linear {
    fn commit_change(&mut self, learning_rate: f32) -> Result<(), Error> {
        if !self.has_pending_change {
            return Ok(());
        }

        {
            let tmp = &mut self.tmp;
            let addition = &mut self.addition;
            let weights_delta = &self.weights_delta;
            let op_result = weights_delta.scalar_mul(-learning_rate, tmp);
            op_result.expect("Ok");
            let weights = &self.weights;
            let op_result = weights.add(&tmp, addition);
            op_result.expect("Ok");
            let weights = &mut self.weights;
            swap(weights, addition);
        }

        {
            let tmp = &mut self.tmp;
            let addition = &mut self.addition;
            let biases_delta = &self.biases_delta;
            let op_result = biases_delta.scalar_mul(-learning_rate, tmp);
            op_result.expect("Ok");
            let biases = &self.biases;
            let op_result = biases.add(&tmp, addition);
            op_result.expect("Ok");
            let biases = &mut self.biases;
            swap(biases, addition);
        }

        self.has_pending_change = false;
        Ok(())
    }

    fn forward(&mut self, input: &Tensor, output: &mut Tensor) -> Result<(), Error> {
        // Use the same convention that is used in tensorflow:
        // y = x @ W^T+b
        // Weights is on the right.
        // W is transposed.
        // X is not transposed.
        let weights = &self.weights;
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

        let biases = &self.biases;
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

    fn backward(&self, layer_delta: &Tensor, previous_layer_delta: &mut Tensor) {
        let layer_weights = &self.weights;

        let op_result = Tensor::matmul(
            layer_weights,
            layer_delta,
            previous_layer_delta,
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

    fn plan_change(&mut self, previous_activation: &Tensor, layer_delta: &Tensor) {
        let weights_delta = &mut self.weights_delta;
        let op_result = Tensor::matmul(
            previous_activation,
            layer_delta,
            weights_delta,
            TRANSPOSE_LHS | TRANSPOSE_RESULT,
        );
        op_result.expect("Ok");

        let biases_delta = &mut self.biases_delta;
        biases_delta.assign(layer_delta);

        self.has_pending_change = true;
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
