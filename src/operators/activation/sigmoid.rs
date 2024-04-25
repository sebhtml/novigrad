use crate::devices::Device;
use crate::{ActivationFunction, DeltaWorkingMemory, OperatorTrait, Tensor};
use crate::{Error, LearningTensor};
use std::cell::RefCell;
use std::f32::consts::E;
use std::ops::Deref;
use std::rc::Rc;

#[derive(Clone)]
pub struct Sigmoid {
    output: Rc<RefCell<Tensor>>,
}

impl Sigmoid {
    pub fn new(device: &Device) -> Self {
        let output = device.tensor(0, 0, vec![]);
        Self {
            output: Rc::new(RefCell::new(output)),
        }
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
        inputs: &Vec<Rc<RefCell<Tensor>>>,
        output: &Rc<RefCell<Tensor>>,
        back_propagated_delta: &Rc<RefCell<Tensor>>,
    ) -> Result<(Rc<RefCell<Tensor>>, Vec<LearningTensor>), Error> {
        let back_propagated_delta: &Tensor = &back_propagated_delta.deref().borrow();
        let mut gradient = device.tensor(0, 0, vec![]);
        {
            // Compute activation function derivative.
            let input: &Tensor = &inputs[0].deref().borrow();
            let output: &Tensor = &output.deref().borrow();
            let layer_f_derivative = &mut error_working_memory.layer_f_derivative;
            self.derive(input, output, layer_f_derivative)?;
            layer_f_derivative.element_wise_mul(device, back_propagated_delta, &mut gradient)?;
        }

        Ok((Rc::new(RefCell::new(gradient)), vec![]))
    }

    fn forward(
        &self,
        _device: &Device,
        inputs: &Vec<Rc<RefCell<Tensor>>>,
    ) -> Result<Rc<RefCell<Tensor>>, Error> {
        {
            let input: &Tensor = &inputs[0].deref().borrow();
            let output: &mut Tensor = &mut self.output.deref().borrow_mut();
            self.activate(input, output)?;
        }
        Ok(self.output.clone())
    }

    fn name(&self) -> &str {
        "Sigmoid"
    }
}
