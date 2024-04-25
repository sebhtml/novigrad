use std::{cell::RefCell, ops::Deref, rc::Rc};

use crate::{devices::Device, DeltaWorkingMemory, Error, LearningTensor, OperatorTrait, Tensor};

use super::LossFunction;

#[cfg(test)]
mod tests;

#[derive(Clone)]
pub struct ResidualSumOfSquares {}

impl Default for ResidualSumOfSquares {
    fn default() -> Self {
        Self {}
    }
}

impl LossFunction for ResidualSumOfSquares {
    /// RSS = Σ (y_i - f(x_i))^2
    fn evaluate(&self, device: &Device, expected: &Tensor, actual: &Tensor) -> Result<f32, Error> {
        if expected.shape() != actual.shape() {
            return Err(Error::IncompatibleTensorShapes);
        }
        let mut diffs = device.tensor(0, 0, vec![]);
        diffs.assign(device, expected);
        Tensor::sub(device, actual, &mut diffs)?;
        Tensor::dot_product(device, &diffs, &diffs)
    }

    fn derive(
        &self,
        device: &Device,
        expected: &Tensor,
        actual: &Tensor,
        result: &mut Tensor,
    ) -> Result<(), Error> {
        result.assign(device, expected);
        Tensor::sub(device, actual, result)?;
        Tensor::scalar_mul(device, -2.0, result);
        Ok(())
    }
}

impl OperatorTrait for ResidualSumOfSquares {
    fn backward(
        &self,
        device: &Device,
        _error_working_memory: &mut DeltaWorkingMemory,
        inputs: &Vec<Rc<RefCell<Tensor>>>,
        _output: &Rc<RefCell<Tensor>>,
        _back_propagated_delta: &Rc<RefCell<Tensor>>,
    ) -> Result<(Rc<RefCell<Tensor>>, Vec<LearningTensor>), Error> {
        debug_assert_eq!(inputs.len(), 2);
        let expected: &Tensor = &inputs[0].deref().borrow();
        let actual: &Tensor = &inputs[1].deref().borrow();
        let mut gradient = device.tensor(0, 0, vec![]);
        self.derive(device, expected, actual, &mut gradient)?;

        Ok((Rc::new(RefCell::new(gradient)), vec![]))
    }

    fn forward(
        &self,
        device: &Device,
        inputs: &Vec<Rc<RefCell<Tensor>>>,
    ) -> Result<Rc<RefCell<Tensor>>, Error> {
        debug_assert_eq!(inputs.len(), 2);
        let expected: &Tensor = &inputs[0].deref().borrow();
        let actual: &Tensor = &inputs[1].deref().borrow();
        let loss = self.evaluate(device, expected, actual)?;
        let output = device.tensor(1, 1, vec![loss]);
        Ok(Rc::new(RefCell::new(output)))
    }

    fn name(&self) -> &str {
        "ResidualSumOfSquares"
    }
}
