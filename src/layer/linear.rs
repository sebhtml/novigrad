use std::mem::swap;

use rand::{distributions::Uniform, thread_rng, Rng};

use crate::{
    ActivationFunction, Error, Layer, Tensor, TRANSPOSE_LHS, TRANSPOSE_RESULT, TRANSPOSE_RHS,
};

pub struct Linear {
    weights: Tensor,
    activation: Box<dyn ActivationFunction>,
}

impl Linear {
    pub fn new(rows: usize, cols: usize, activation: Box<dyn ActivationFunction>) -> Self {
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
        Linear {
            weights,
            activation: activation,
        }
    }

    fn weights<'a>(&'a self) -> &'a Tensor {
        &self.weights
    }
}

impl Layer for Linear {
    fn apply_weight_deltas(
        &mut self,
        addition: &mut Tensor,
        weight_deltas: &Tensor,
    ) -> Result<(), Error> {
        {
            let weights = &self.weights;
            let op_result = weights.sub(weight_deltas, addition);
            op_result.expect("Ok");
        }

        let weights = &mut self.weights;
        swap(weights, addition);
        Ok(())
    }

    fn activation<'a>(&'a self) -> &'a Box<dyn ActivationFunction> {
        &self.activation
    }

    fn forward(
        &self,
        input: &Tensor,
        matrix_product: &mut Tensor,
        activation_tensor: &mut Tensor,
    ) -> Result<(), Error> {
        let weights = &self.weights;
        let op_result = Tensor::matmul(input, weights, matrix_product, TRANSPOSE_RHS);
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
        let activation_function = &self.activation;
        let op_result = activation_function.activate(&matrix_product, activation_tensor);
        op_result.expect("Ok");
        Ok(())
    }

    fn backward(&self, layer_delta: &Tensor, output_diff: &mut Tensor) {
        let layer_weights = &self.weights;

        let op_result = Tensor::matmul(
            layer_weights,
            layer_delta,
            output_diff,
            TRANSPOSE_LHS | TRANSPOSE_RHS | TRANSPOSE_RESULT,
        );

        op_result.expect("Ok");
    }
}
