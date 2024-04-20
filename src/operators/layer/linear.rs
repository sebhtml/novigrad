use std::{cell::RefCell, ops::Deref, rc::Rc};

use rand::{distributions::Uniform, thread_rng, Rng};

use crate::{accelerator::Accelerator, DeltaWorkingMemory, Error, Gradient, OperatorTrait, Tensor};

pub struct Linear {
    weights: Rc<RefCell<Tensor>>,
    biases: Rc<RefCell<Tensor>>,
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
            weights: Rc::new(RefCell::new(weights)),
            biases: Rc::new(RefCell::new(biases)),
        }
    }
}

impl OperatorTrait for Linear {
    fn forward(
        &mut self,
        accelerator: &Accelerator,
        inputs: &Vec<Tensor>,
        output: &mut Tensor,
    ) -> Result<(), Error> {
        debug_assert_eq!(inputs.len(), 1);
        let input = &inputs[0];
        // Use the same convention that is used in tensorflow:
        // Y = X @ W^T + B
        // Weights is on the right.
        // X is not transposed.
        // W is transposed.

        // use GEMM to do C = A * W^T + C  with weights and biases all together.
        let weights: &Tensor = &self.weights.deref().borrow();
        let biases: &Tensor = &self.biases.deref().borrow();
        let a = input;
        let b = weights;
        let c = output;
        c.assign(accelerator, biases);
        let op_result = Tensor::gemm(accelerator, false, true, 1.0, a, b, 1.0, c, false);
        match op_result {
            Ok(_) => (),
            Err(_) => {
                let mut w_t = Tensor::default();
                b.transpose(&mut w_t);
                println!("Incompatible shapes in matrix multiplication");
                println!("Between X {:?} and W^T {:?}", input.shape(), w_t.shape(),);
                debug_assert!(false);
            }
        }

        Ok(())
    }

    fn backward(
        &self,
        _inputs: &Vec<Tensor>,
        accelerator: &Accelerator,
        layer_output_delta: &Tensor,
        previous_layer_output_delta: &mut Tensor,
    ) {
        let weights: &Tensor = &self.weights.deref().borrow();
        let a = weights;
        let b = layer_output_delta;
        let c = previous_layer_output_delta;
        c.reset(b.rows(), a.cols(), 0.0);
        let op_result = Tensor::matmul(accelerator, true, true, a, b, c, true);

        op_result.expect("Ok");
    }

    fn get_layer_output_delta(
        &self,
        accelerator: &Accelerator,
        _working_memory: &mut DeltaWorkingMemory,
        _inputs: &Vec<Tensor>,
        _layer_output: &Tensor,
        back_propagated_delta: &Tensor,
        layer_delta: &mut Tensor,
    ) {
        layer_delta.assign(accelerator, back_propagated_delta)
    }

    fn compute_gradients(
        &mut self,
        accelerator: &Accelerator,
        inputs: &Vec<Tensor>,
        layer_output_delta: &Tensor,
    ) -> Result<Vec<Gradient>, Error> {
        let mut gradients = vec![];
        let mut weights_gradient = Tensor::default();
        let mut biases_gradient = Tensor::default();
        let layer_input = &inputs[0];
        let a = layer_input;
        let b = layer_output_delta;
        let c = &mut weights_gradient;
        c.reset(b.cols(), a.cols(), 0.0);
        let op_result = Tensor::matmul(accelerator, true, false, a, b, c, true);
        op_result.expect("Ok");

        biases_gradient.assign(accelerator, layer_output_delta);

        gradients.push(Gradient::new(self.weights.clone(), weights_gradient));
        gradients.push(Gradient::new(self.biases.clone(), biases_gradient));

        Ok(gradients)
    }

    fn name(&self) -> &str {
        "Linear"
    }
}
