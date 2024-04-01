use std::mem::swap;

use rand::{distributions::Uniform, thread_rng, Rng};

use crate::{
    ActivationFunction, DeltaWorkingMemory, Error, Layer, Tensor, TRANSPOSE_LHS, TRANSPOSE_RESULT,
    TRANSPOSE_RHS,
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

    fn activation<'a>(&'a self) -> &'a Box<dyn ActivationFunction> {
        &self.activation
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

    fn get_layer_delta(
        &self,
        working_memory: &mut DeltaWorkingMemory,
        layer_product_tensor: &Tensor,
        layer_activation_tensor: &Tensor,
        next_layer: Option<&Box<dyn Layer>>,
        next_layer_delta: &Tensor,
        using_softmax_and_cross_entropy_loss: bool,
        layer_delta: &mut Tensor,
    ) {
        let layer_f_derivative = &mut working_memory.layer_f_derivative;
        let layer_activation_function = &self.activation;
        let output_diff = &mut working_memory.output_diff;

        match next_layer {
            None => {
                // use the output of the loss function.
                output_diff.assign(next_layer_delta);
            }
            Some(next_layer) => {
                // Hidden layer
                next_layer.backward(next_layer_delta, output_diff);
            }
        }

        // Compute activation function derivative.
        let is_last_layer = next_layer.is_none();
        if is_last_layer && using_softmax_and_cross_entropy_loss {
            layer_activation_tensor
                .scalar_add(1.0, layer_f_derivative)
                .expect("Ok");
        } else {
            let op_result = layer_activation_function.derive(
                layer_product_tensor,
                layer_activation_tensor,
                layer_f_derivative,
            );
            op_result.expect("Ok");
        }

        let op_result = layer_f_derivative.element_wise_mul(output_diff, layer_delta);
        op_result.expect("Ok");
    }
}
