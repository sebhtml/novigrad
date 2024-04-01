use std::{mem::swap, rc::Rc};

use rand::{distributions::Uniform, thread_rng, Rng};

use crate::{ActivationFunction, Error, Layer, Tensor, TRANSPOSE_RHS};

pub struct Linear {
    weights: Tensor,
    activation: Rc<dyn ActivationFunction>,
}

impl Linear {
    pub fn new(rows: usize, cols: usize, activation: Rc<dyn ActivationFunction>) -> Self {
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
}
impl Layer for Linear {
    fn weights<'a>(&'a self) -> &'a Tensor {
        &self.weights
    }

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

    fn activation(&self) -> Rc<dyn ActivationFunction> {
        self.activation.clone()
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
}
