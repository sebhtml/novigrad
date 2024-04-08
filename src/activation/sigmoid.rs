use crate::{ActivationFunction, Layer, Tensor};
use crate::{Error, TensorTrait};
use std::f32::consts::E;

#[derive(Clone, Default)]
pub struct SigmoidConfig {}

#[derive(Clone, Default)]
pub struct Sigmoid {}

impl Into<Sigmoid> for &SigmoidConfig {
    fn into(self) -> Sigmoid {
        Default::default()
    }
}

impl Sigmoid {
    fn activate_f(product_matrix: &Tensor, result: &mut Tensor) -> Result<(), Error> {
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

    fn derive_f(
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

impl ActivationFunction for Sigmoid {
    fn activate(&self, product_matrix: &Tensor, result: &mut Tensor) -> Result<(), Error> {
        Self::activate_f(product_matrix, result)
    }

    fn derive(
        &self,
        product_matrix: &Tensor,
        activation_matrix: &Tensor,
        result: &mut Tensor,
    ) -> Result<(), Error> {
        Self::derive_f(product_matrix, activation_matrix, result)
    }
}

// TODO refactor this since it is a copy-and-paste of Softmax impl.
impl Layer for Sigmoid {
    fn plan_change(
        &mut self,
        _learning_rate: f32,
        _previous_activation: &Tensor,
        _layer_delta: &Tensor,
    ) {
    }

    fn commit_change(&mut self) -> Result<(), Error> {
        Ok(())
    }

    fn forward(&mut self, input: &Tensor, output: &mut Tensor) -> Result<(), Error> {
        Self::activate_f(input, output)
    }

    fn backward(&self, layer_delta: &Tensor, output_diff: &mut Tensor) {
        output_diff.assign(layer_delta)
    }

    fn get_layer_delta(
        &self,
        working_memory: &mut crate::DeltaWorkingMemory,
        layer_input: &Tensor,
        layer_output: &Tensor,
        next_layer: Option<&crate::LayerType>,
        next_layer_delta: &Tensor,
        layer_delta: &mut Tensor,
    ) {
        let output_diff = &mut working_memory.output_diff;

        match next_layer {
            None => {
                // use the output of the loss function.
                output_diff.assign(next_layer_delta);
            }
            Some(next_layer) => {
                // Hidden layer
                next_layer.backward(next_layer_delta, output_diff);
            }
        }

        // Compute activation function derivative.
        let layer_f_derivative = &mut working_memory.layer_f_derivative;
        let op_result = Self::derive_f(layer_input, layer_output, layer_f_derivative);
        op_result.expect("Ok");
        let op_result = layer_f_derivative.element_wise_mul(output_diff, layer_delta);
        op_result.expect("Ok");
    }
}
