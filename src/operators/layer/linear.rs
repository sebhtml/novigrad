use rand::{distributions::Uniform, thread_rng, Rng};

use crate::{
    accelerator::Accelerator, DeltaWorkingMemory, DifferentiableTensor, Error, OperatorTrait,
    Tensor,
};

pub struct Linear {
    weights: DifferentiableTensor,
    biases: DifferentiableTensor,
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
        }
    }
}

impl OperatorTrait for Linear {
    fn commit_change(
        &mut self,
        accelerator: &Accelerator,
        learning_rate: f32,
    ) -> Result<(), Error> {
        self.weights.commit_change(accelerator, learning_rate);
        self.biases.commit_change(accelerator, learning_rate);
        Ok(())
    }

    fn forward(
        &mut self,
        accelerator: &Accelerator,
        input: &Tensor,
        output: &mut Tensor,
    ) -> Result<(), Error> {
        // Use the same convention that is used in tensorflow:
        // Y = X @ W^T + B
        // Weights is on the right.
        // X is not transposed.
        // W is transposed.

        // use GEMM to do C = A * W^T + C  with weights and biases all together.
        let biases = &self.biases.tensor;
        let a = input;
        let b = &self.weights.tensor;
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
        accelerator: &Accelerator,
        layer_output_delta: &Tensor,
        previous_layer_output_delta: &mut Tensor,
    ) {
        let a = &self.weights.tensor;
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
        _layer_input: &Tensor,
        _layer_output: &Tensor,
        back_propagated_delta: &Tensor,
        _is_last_layer: bool,
        layer_delta: &mut Tensor,
    ) {
        layer_delta.assign(accelerator, back_propagated_delta)
    }

    fn compute_gradient(
        &mut self,
        accelerator: &Accelerator,
        layer_input: &Tensor,
        layer_output_delta: &Tensor,
    ) {
        let a = layer_input;
        let b = layer_output_delta;
        let c = &mut self.weights.gradient;
        c.reset(b.cols(), a.cols(), 0.0);
        let op_result = Tensor::matmul(accelerator, true, false, a, b, c, true);
        op_result.expect("Ok");
        self.weights.has_gradient = true;

        self.biases.gradient.assign(accelerator, layer_output_delta);
        self.biases.has_gradient = true;
    }

    fn forward2(
        &mut self,
        _accelerator: &Accelerator,
        _input1: &Tensor,
        _input2: &Tensor,
    ) -> Result<Tensor, Error> {
        panic!()
    }
}
