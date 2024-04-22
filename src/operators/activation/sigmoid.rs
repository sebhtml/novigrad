use crate::devices::Device;
use crate::{ActivationFunction, DeltaWorkingMemory, OperatorTrait, Tensor};
use crate::{Error, Gradient};
use std::f32::consts::E;
use std::rc::Rc;

#[derive(Clone, Default)]
pub struct Sigmoid {}

impl ActivationFunction for Sigmoid {
    fn activate(&self, product_matrix: &Tensor, result: &mut Tensor) -> Result<(), Error> {
        result.reset(
            product_matrix.rows(),
            product_matrix.cols(),
            Default::default(),
        );
        let rows = product_matrix.rows();
        let cols = product_matrix.cols();
        let mut row = 0;
        while row < rows {
            let mut col = 0;
            while col < cols {
                let x = product_matrix.get(row, col);
                let y = 1.0 / (1.0 + E.powf(-x));
                result.set(row, col, y);
                col += 1;
            }
            row += 1;
        }
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
        let mut row = 0;
        while row < rows {
            let mut col = 0;
            while col < cols {
                let x = activation_matrix.get(row, col);
                let y = x * (1.0 - x);
                result.set(row, col, y);
                col += 1;
            }
            row += 1;
        }
        Ok(())
    }
}

impl OperatorTrait for Sigmoid {
    fn backward(
        &self,
        device: &Device,
        error_working_memory: &mut DeltaWorkingMemory,
        inputs: &Vec<Rc<Tensor>>,
        output: &Rc<Tensor>,
        back_propagated_delta: &mut Tensor,
        layer_delta: &mut Tensor,
    ) -> Result<(Tensor, Vec<Gradient>), Error> {
        {
            // Compute activation function derivative.
            let input = &inputs[0];
            let layer_f_derivative = &mut error_working_memory.layer_f_derivative;
            self.derive(input, output, layer_f_derivative)?;
            layer_f_derivative.element_wise_mul(back_propagated_delta, layer_delta)?;
        }

        back_propagated_delta.assign(device, layer_delta);

        Ok((back_propagated_delta.clone(), vec![]))
    }

    fn forward(&self, _device: &Device, inputs: &Vec<Rc<Tensor>>) -> Result<Rc<Tensor>, Error> {
        let input = &inputs[0];
        let mut output = Tensor::new(0, 0, vec![0.0]);
        self.activate(input, &mut output)?;
        Ok(Rc::new(output))
    }

    fn name(&self) -> &str {
        "Sigmoid"
    }
}
