use super::LossFunction;
use crate::{accelerator::Accelerator, Error, OperatorTrait, Tensor};

#[derive(Clone)]
pub struct CrossEntropyLoss {}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self {}
    }
}

const EPSILON: f32 = 1e-8;

impl LossFunction for CrossEntropyLoss {
    /// H(P, Q) = - Σ (P(i) * log(Q(i)))
    fn evaluate(
        &self,
        _accelerator: &Accelerator,
        expected: &Tensor,
        actual: &Tensor,
    ) -> Result<f32, Error> {
        debug_assert_eq!(actual.shape(), expected.shape());
        let p = expected;
        let q = actual;
        if p.shape() != q.shape() {
            return Err(Error::IncompatibleTensorShapes);
        }
        let rows = p.rows();
        debug_assert_eq!(rows, 1);
        let cols = p.cols();
        let mut col = 0;
        let mut sum = 0.0;
        while col < cols {
            let p_i = p.get(0, col);
            let q_i = q.get(0, col) + EPSILON;
            sum += p_i * f32::ln(q_i);
            col += 1;
        }
        debug_assert!(sum.is_finite());
        Ok(-sum)
    }

    /// When Cross-Entropy Loss is used with a Softmax activation function,
    /// then we don't need to derive the softmax activations.
    /// The derivative of the Loss in respect to logits (before activation) is
    /// output of the softmax function - expected output (one-hot encoded)
    fn derive(
        &self,
        accelerator: &Accelerator,
        expected: &Tensor,
        actual: &Tensor,
        result: &mut Tensor,
    ) -> Result<(), Error> {
        result.assign(accelerator, actual);
        Tensor::sub(accelerator, expected, result)
    }
}

impl OperatorTrait for CrossEntropyLoss {
    fn compute_gradient(
        &mut self,
        _accelerator: &Accelerator,
        _layer_input: &Tensor,
        _layer_output_delta: &Tensor,
    ) {
    }

    fn commit_change(
        &mut self,
        _accelerator: &Accelerator,
        _learning_rate: f32,
    ) -> Result<(), Error> {
        Ok(())
    }

    fn forward(
        &mut self,
        accelerator: &Accelerator,
        inputs: &Vec<Tensor>,
        output: &mut Tensor,
    ) -> Result<(), Error> {
        debug_assert_eq!(inputs.len(), 2);
        let expected = &inputs[0];
        let actual = &inputs[0];
        let loss = self.evaluate(accelerator, expected, actual)?;
        let tensor = Tensor::new(1, 1, vec![loss]);
        output.assign(accelerator, &tensor);
        Ok(())
    }

    fn backward(
        &self,
        inputs: &Vec<Tensor>,
        accelerator: &Accelerator,
        _layer_output_delta: &Tensor,
        previous_layer_output_delta: &mut Tensor,
    ) {
        debug_assert_eq!(inputs.len(), 2);
        let expected = &inputs[0];
        let actual = &inputs[0];
        let op_result = self.derive(accelerator, expected, actual, previous_layer_output_delta);
        op_result.expect("Ok");
    }

    fn get_layer_output_delta(
        &self,
        accelerator: &Accelerator,
        _working_memory: &mut crate::DeltaWorkingMemory,
        _inputs: &Vec<Tensor>,
        _layer_output: &Tensor,
        back_propagated_layer_output_delta: &Tensor,
        _is_last_layer: bool,
        layer_output_delta: &mut Tensor,
    ) {
        layer_output_delta.assign(accelerator, back_propagated_layer_output_delta)
    }
}
