use std::{cell::RefCell, ops::Deref, rc::Rc};

use rand::{distributions::Uniform, thread_rng, Rng};

use crate::{devices::Device, DeltaWorkingMemory, Error, Gradient, OperatorTrait, Tensor};

pub struct Linear {
    weights: Rc<RefCell<Tensor>>,
    biases: Rc<RefCell<Tensor>>,
}

impl Linear {
    pub fn new(
        weights_rows: usize,
        weights_cols: usize,
        bias_rows: usize,
        device: &Device,
    ) -> Self {
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
        let weights = device.tensor(weights_rows, weights_cols, weights);

        let mut biases = device.tensor(0, 0, vec![]);
        biases.reset(bias_rows, weights_rows, Default::default());

        Linear {
            weights: Rc::new(RefCell::new(weights)),
            biases: Rc::new(RefCell::new(biases)),
        }
    }
}

impl OperatorTrait for Linear {
    fn forward(
        &self,
        device: &Device,
        inputs: &Vec<Rc<RefCell<Tensor>>>,
    ) -> Result<Rc<RefCell<Tensor>>, Error> {
        debug_assert_eq!(inputs.len(), 1);
        let input: &Tensor = &inputs[0].deref().borrow();
        let mut output = device.tensor(0, 0, vec![]);
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
        let c = &mut output;
        c.assign(device, biases);
        let op_result = Tensor::gemm(device, false, true, 1.0, a, b, 1.0, c, false);
        match op_result {
            Ok(_) => (),
            Err(_) => {
                let mut w_t = device.tensor(0, 0, vec![]);
                b.transpose(&mut w_t);
                println!("Incompatible shapes in matrix multiplication");
                println!("Between X {:?} and W^T {:?}", input.shape(), w_t.shape(),);
                debug_assert!(false);
            }
        }

        Ok(Rc::new(RefCell::new(output)))
    }

    fn backward(
        &self,
        device: &Device,
        _error_working_memory: &mut DeltaWorkingMemory,
        inputs: &Vec<Rc<RefCell<Tensor>>>,
        _output: &Rc<RefCell<Tensor>>,
        back_propagated_delta: &mut Tensor,
        layer_delta: &mut Tensor,
    ) -> Result<(Tensor, Vec<Gradient>), Error> {
        {
            layer_delta.assign(device, back_propagated_delta);
        }

        let mut gradients = vec![];
        {
            let mut weights_gradient = device.tensor(0, 0, vec![]);
            let mut biases_gradient = device.tensor(0, 0, vec![]);
            let input: &Tensor = &inputs[0].deref().borrow();
            let a: &Tensor = input;
            let b: &Tensor = layer_delta;
            let c: &mut Tensor = &mut weights_gradient;
            c.reset(b.cols(), a.cols(), 0.0);
            let op_result = Tensor::matmul(device, true, false, a, b, c, true);
            op_result.expect("Ok");

            biases_gradient.assign(device, layer_delta);

            gradients.push(Gradient::new(self.weights.clone(), weights_gradient));
            gradients.push(Gradient::new(self.biases.clone(), biases_gradient));
        }

        {
            let weights: &Tensor = &self.weights.deref().borrow();
            let a: &Tensor = weights;
            let b: &Tensor = layer_delta;
            let c: &mut Tensor = back_propagated_delta;
            c.reset(b.rows(), a.cols(), 0.0);
            Tensor::matmul(device, true, true, a, b, c, true)?;
        }

        Ok((back_propagated_delta.clone(), gradients))
    }

    fn name(&self) -> &str {
        "Linear"
    }
}
