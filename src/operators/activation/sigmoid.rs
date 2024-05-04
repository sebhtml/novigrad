use crate::devices::Device;
use crate::{ActivationFunction, OperatorTrait, TensorF32};
use crate::{Error, Tensor};
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
    fn activate(&self, product_matrix: &TensorF32, result: &mut TensorF32) -> Result<(), Error> {
        let rows = product_matrix.rows();
        let cols = product_matrix.cols();
        let values = product_matrix.get_values()?;
        let mut result_values = result.get_values()?;
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
        _product_matrix: &TensorF32,
        activation_matrix: &TensorF32,
        result: &mut TensorF32,
    ) -> Result<(), Error> {
        let rows = activation_matrix.rows();
        let cols = activation_matrix.cols();
        let values = activation_matrix.get_values()?;
        let mut result_values = result.get_values()?;
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
    fn backward(&self, device: &Device, inputs: &[Tensor], output: &Tensor) -> Result<(), Error> {
        let output_gradient: &TensorF32 = &output.gradient().deref().borrow();
        let backward_gradient: &mut TensorF32 = &mut inputs[0].gradient().deref().borrow_mut();
        // Compute activation function derivative.
        let input: &TensorF32 = &inputs[0].tensor().deref().borrow();
        let output: &TensorF32 = &output.tensor().deref().borrow();
        let rows = input.rows();
        let cols = input.cols();
        let len = rows * cols;
        let mut layer_f_derivative = device.tensor_f32(rows, cols, vec![0.0; len]);
        self.derive(input, output, &mut layer_f_derivative)?;
        layer_f_derivative.element_wise_mul(device, output_gradient, backward_gradient)?;
        Ok(())
    }

    fn forward(&self, device: &Device, inputs: &[Tensor]) -> Result<Tensor, Error> {
        let input: &TensorF32 = &inputs[0].tensor().deref().borrow();
        let rows = input.rows();
        let cols = input.cols();
        let len = rows * cols;
        let output = device.tensor(inputs, rows, cols, vec![0.0; len], false);
        {
            let output: &mut TensorF32 = &mut output.tensor().deref().borrow_mut();
            self.activate(input, output)?;
        }
        Ok(output)
    }

    fn name(&self) -> &str {
        "Sigmoid"
    }
}
