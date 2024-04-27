use crate::devices::Device;
use crate::{ActivationFunction, DeltaWorkingMemory, OperatorTrait, Tensor};
use crate::{Error, LearningTensor};
use std::f32::consts::E;
use std::ops::Deref;

#[derive(Clone)]
pub struct Sigmoid {}

impl Sigmoid {
    pub fn new(_device: &Device) -> Self {
        Self {}
    }
}

impl ActivationFunction for Sigmoid {
    fn activate(&self, product_matrix: &Tensor, result: &mut Tensor) -> Result<(), Error> {
        result.reset(
            product_matrix.rows(),
            product_matrix.cols(),
            Default::default(),
        );
        let rows = product_matrix.rows();
        let cols = product_matrix.cols();
        let values = product_matrix.get_values();
        let mut result_values = result.get_values();
        let mut row = 0;
        while row < rows {
            let mut col = 0;
            while col < cols {
                let x = values[product_matrix.index(row, col)];
                let y = 1.0 / (1.0 + E.powf(-x));
                result_values[result.index(row, col)] = y;
                col += 1;
            }
            row += 1;
        }
        result.set_values(result_values);
        Ok(())
    }

    fn derive(
        &self,
        _product_matrix: &Tensor,
        activation_matrix: &Tensor,
        result: &mut Tensor,
    ) -> Result<(), Error> {
        result.reset(
            activation_matrix.rows(),
            activation_matrix.cols(),
            Default::default(),
        );
        let rows = activation_matrix.rows();
        let cols = activation_matrix.cols();
        let values = activation_matrix.get_values();
        let mut result_values = result.get_values();
        let mut row = 0;
        while row < rows {
            let mut col = 0;
            while col < cols {
                let x = values[activation_matrix.index(row, col)];
                let y = x * (1.0 - x);
                result_values[result.index(row, col)] = y;
                col += 1;
            }
            row += 1;
        }

        result.set_values(result_values);
        Ok(())
    }
}

impl OperatorTrait for Sigmoid {
    fn backward(
        &self,
        device: &Device,
        error_working_memory: &mut DeltaWorkingMemory,
        inputs: &Vec<LearningTensor>,
        output: &LearningTensor,
    ) -> Result<(), Error> {
        let back_propagated_delta: &Tensor = &output.gradient().deref().borrow();
        let backward_gradient: &mut Tensor = &mut inputs[0].gradient().deref().borrow_mut();
        // Compute activation function derivative.
        let input: &Tensor = &inputs[0].tensor().deref().borrow();
        let output: &Tensor = &output.tensor().deref().borrow();
        let layer_f_derivative = &mut error_working_memory.layer_f_derivative;
        self.derive(input, output, layer_f_derivative)?;
        layer_f_derivative.element_wise_mul(device, back_propagated_delta, backward_gradient)?;
        Ok(())
    }

    fn forward(
        &self,
        device: &Device,
        inputs: &Vec<LearningTensor>,
    ) -> Result<LearningTensor, Error> {
        let input: &Tensor = &inputs[0].tensor().deref().borrow();
        let output = device.learning_tensor(0, 0, vec![], false);
        {
            let output: &mut Tensor = &mut output.tensor().deref().borrow_mut();
            self.activate(input, output)?;
        }
        Ok(output)
    }

    fn name(&self) -> &str {
        "Sigmoid"
    }
}
