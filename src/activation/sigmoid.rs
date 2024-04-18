use crate::accelerator::Accelerator;
use crate::Error;
use crate::{ActivationFunction, DeltaWorkingMemory, OperatorTrait, Tensor};
use std::f32::consts::E;

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
        _accelerator: &Accelerator,
        input: &Tensor,
        output: &mut Tensor,
    ) -> Result<(), Error> {
        self.activate(input, output)
    }

    fn backward(
        &self,
        accelerator: &Accelerator,
        layer_delta: &Tensor,
        previous_layer_delta: &mut Tensor,
    ) {
        previous_layer_delta.assign(accelerator, layer_delta)
    }

    fn get_layer_output_delta(
        &self,
        _accelerator: &Accelerator,
        working_memory: &mut DeltaWorkingMemory,
        layer_input: &Tensor,
        layer_output: &Tensor,
        back_propagated_delta: &Tensor,
        _is_last_layer: bool,
        layer_delta: &mut Tensor,
    ) {
        // Compute activation function derivative.
        let layer_f_derivative = &mut working_memory.layer_f_derivative;
        let op_result = self.derive(layer_input, layer_output, layer_f_derivative);
        op_result.expect("Ok");
        let op_result = layer_f_derivative.element_wise_mul(back_propagated_delta, layer_delta);
        op_result.expect("Ok");
    }
}
