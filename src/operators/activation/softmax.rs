use crate::devices::Device;
use crate::{ActivationFunction, DeltaWorkingMemory, OperatorTrait, Tensor};
use crate::{Error, LearningTensor};
use std::f32::consts::E;
use std::ops::Deref;

#[derive(Clone)]
pub struct Softmax {
    using_cross_entropy_loss: bool,
}

impl Softmax {
    pub fn new(using_cross_entropy_loss: bool, _device: &Device) -> Self {
        Self {
            using_cross_entropy_loss,
        }
    }
}

impl ActivationFunction for Softmax {
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

impl OperatorTrait for Softmax {
    fn backward(
        &self,
        device: &Device,
        error_working_memory: &mut DeltaWorkingMemory,
        inputs: &Vec<LearningTensor>,
        output: &LearningTensor,
        _enabled_gradients: &mut Vec<LearningTensor>,
    ) -> Result<(), Error> {
        let back_propagated_delta: &Tensor = &output.gradient().deref().borrow();
        {
            let backward_gradient: &mut Tensor = &mut inputs[0].gradient().deref().borrow_mut();
            // Compute activation function derivative.
            if self.using_cross_entropy_loss {
                // Softmax and Cross Entropy Loss are best friends.
                backward_gradient.assign(device, back_propagated_delta);
            } else {
                let input: &Tensor = &inputs[0].tensor().deref().borrow();
                let output: &Tensor = &output.tensor().deref().borrow();
                let layer_f_derivative = &mut error_working_memory.layer_f_derivative;
                self.derive(input, output, layer_f_derivative)?;

                layer_f_derivative.element_wise_mul(
                    device,
                    back_propagated_delta,
                    backward_gradient,
                )?;
            }
        }

        Ok(())
    }

    fn forward(
        &self,
        device: &Device,
        inputs: &Vec<LearningTensor>,
    ) -> Result<LearningTensor, Error> {
        let output = device.learning_tensor(0, 0, vec![]);
        {
            let input: &Tensor = &inputs[0].tensor().deref().borrow();
            let output: &mut Tensor = &mut output.tensor().deref().borrow_mut();
            self.activate(input, output)?;
        }
        Ok(output)
    }

    fn name(&self) -> &str {
        "Softmax"
    }
}
