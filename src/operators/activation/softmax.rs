use crate::devices::Device;
use crate::{ActivationFunction, OperatorTrait, TensorF32};
use crate::{Error, Tensor};
use std::f32::consts::E;
use std::ops::Deref;
use std::rc::Rc;

#[derive(Clone)]
pub struct Softmax {
    device: Device,
    using_cross_entropy_loss: bool,
}

impl Softmax {
    pub fn new(using_cross_entropy_loss: bool, device: &Device) -> Self {
        Self {
            device: device.clone(),
            using_cross_entropy_loss,
        }
    }
}

impl ActivationFunction for Softmax {
    fn activate(&self, product_matrix: &TensorF32, result: &mut TensorF32) -> Result<(), Error> {
        let rows = product_matrix.rows();
        let cols = product_matrix.cols();
        let values = product_matrix.get_values()?;
        let mut result_values = result.get_values()?;
        let mut row = 0;
        while row < rows {
            // Find max

            let mut max = values[product_matrix.index(row, 0)];
            let mut col = 0;
            while col < cols {
                let x = values[product_matrix.index(row, col)];
                max = max.max(x);
                col += 1;
            }

            // For each value:
            // 1. substract the max
            // 2. compute E^x
            // 3. add result to sum
            let mut sum = 0.0;
            let mut col = 0;
            while col < cols {
                let x = values[product_matrix.index(row, col)];
                let y = E.powf(x - max);
                result_values[result.index(row, col)] = y;
                sum += y;
                col += 1;
            }

            // Divide every value by sum.

            let mut col = 0;
            while col < cols {
                let x = result_values[result.index(row, col)];
                let y = x / sum;
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

impl OperatorTrait for Softmax {
    fn backward(&self, inputs: &[Tensor], output: &Tensor) -> Result<(), Error> {
        let output_gradient: &TensorF32 = &output.gradient().deref().borrow();
        let backward_gradient: &mut TensorF32 = &mut inputs[0].gradient().deref().borrow_mut();
        // Compute activation function derivative.
        if self.using_cross_entropy_loss {
            // Softmax and Cross Entropy Loss are best friends.
            TensorF32::copy(output_gradient, backward_gradient)?;
        } else {
            let input: &TensorF32 = &inputs[0].tensor().deref().borrow();
            let output: &TensorF32 = &output.tensor().deref().borrow();
            let rows = input.rows();
            let cols = input.cols();
            let len = rows * cols;
            let mut layer_f_derivative = self.device.tensor_f32(rows, cols, vec![0.0; len]);
            self.derive(input, output, &mut layer_f_derivative)?;

            layer_f_derivative.element_wise_mul(output_gradient, backward_gradient)?;
        }

        Ok(())
    }

    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, Error> {
        let input: &TensorF32 = &inputs[0].tensor().deref().borrow();
        let rows = input.rows();
        let cols = input.cols();
        let len = rows * cols;
        let output = self.device.tensor(
            Rc::new(self.clone()),
            inputs,
            rows,
            cols,
            vec![0.0; len],
            false,
        );
        Ok(output)
    }

    fn name(&self) -> &str {
        "Softmax"
    }

    fn forward_realize(&self, inputs: &[Tensor], output: &Tensor) -> Result<(), Error> {
        let input: &TensorF32 = &inputs[0].tensor().deref().borrow();
        let output: &mut TensorF32 = &mut output.tensor().deref().borrow_mut();
        self.activate(input, output)
    }
}
