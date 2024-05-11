use crate::devices::Device;
use crate::{ActivationFunction, Operator, TensorF32, UnaryOperator};
use crate::{Error, Tensor};
use std::f32::consts::E;
use std::ops::Deref;
use std::rc::Rc;

/// https://onnx.ai/onnx/operators/onnx__Sigmoid.html
#[derive(Clone)]
pub struct Sigmoid {
    device: Device,
}

impl Sigmoid {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
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

impl UnaryOperator for Sigmoid {
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        let input_t: &TensorF32 = &input.tensor().deref().borrow();
        let rows = input_t.rows();
        let cols = input_t.cols();
        let len = rows * cols;
        let output = self.device.tensor(
            Rc::new(self.clone()),
            &[input],
            rows,
            cols,
            vec![0.0; len],
            true,
            false,
        );
        Ok(output)
    }
}

impl Operator for Sigmoid {
    fn name(&self) -> &str {
        "Sigmoid"
    }

    fn forward_realize(&self, inputs: &[&Tensor], output: &Tensor) -> Result<(), Error> {
        let input: &TensorF32 = &inputs[0].tensor().deref().borrow();
        let output: &mut TensorF32 = &mut output.tensor().deref().borrow_mut();
        self.activate(input, output)
    }

    fn backward(&self, inputs: &[&Tensor], output: &Tensor) -> Result<(), Error> {
        if inputs[0].requires_grad() {
            let input_gradient: &mut TensorF32 = &mut inputs[0].gradient().deref().borrow_mut();
            let output_gradient: &TensorF32 = &output.gradient().deref().borrow();
            // Compute activation function derivative.
            let input: &TensorF32 = &inputs[0].tensor().deref().borrow();
            let output: &TensorF32 = &output.tensor().deref().borrow();
            let rows = input.rows();
            let cols = input.cols();
            let len = rows * cols;
            let mut layer_f_derivative = self.device.tensor_f32(rows, cols, vec![0.0; len]);
            self.derive(input, output, &mut layer_f_derivative)?;
            TensorF32::mul(&layer_f_derivative, output_gradient, input_gradient)?;
        }

        Ok(())
    }
}
