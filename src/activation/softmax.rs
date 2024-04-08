use crate::{ActivationFunction, Layer, Tensor};
use crate::{Error, TensorTrait};
use std::f32::consts::E;

#[derive(Clone)]
pub struct SoftmaxConfig {
    pub using_softmax_and_cross_entropy_loss: bool,
}

#[derive(Clone)]
pub struct Softmax {
    using_softmax_and_cross_entropy_loss: bool,
}

impl Into<Softmax> for &SoftmaxConfig {
    fn into(self) -> Softmax {
        Softmax {
            using_softmax_and_cross_entropy_loss: self.using_softmax_and_cross_entropy_loss,
        }
    }
}

impl Softmax {
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
            // Find max

            let mut max = product_matrix.get(row, 0);
            let mut col = 0;
            while col < cols {
                let x = product_matrix.get(row, col);
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
                let x = product_matrix.get(row, col);
                let y = E.powf(x - max);
                result.set(row, col, y);
                sum += y;
                col += 1;
            }

            // Divide every value by sum.

            let mut col = 0;
            while col < cols {
                let x = result.get(row, col);
                let y = x / sum;
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

impl ActivationFunction for Softmax {
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

impl Layer for Softmax {
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
        back_propagated_delta: &Tensor,
        is_last_layer: bool,
        layer_delta: &mut Tensor,
    ) {
        // Compute activation function derivative.
        if is_last_layer && self.using_softmax_and_cross_entropy_loss {
            // Softmax and Cross Entropy Loss are best friends.
            layer_delta.assign(&back_propagated_delta);
        } else {
            let layer_f_derivative = &mut working_memory.layer_f_derivative;
            let op_result = Self::derive_f(layer_input, layer_output, layer_f_derivative);
            op_result.expect("Ok");
            let op_result = layer_f_derivative.element_wise_mul(back_propagated_delta, layer_delta);
            op_result.expect("Ok");
        }
    }
}
