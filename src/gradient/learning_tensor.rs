use std::{cell::RefCell, ops::Deref, rc::Rc};

use crate::{DeltaWorkingMemory, Device, Error, OperatorTrait, Record, Tape, Tensor};

#[derive(Clone)]
pub struct LearningTensor {
    tensor: Rc<RefCell<Tensor>>,
    gradient: Rc<RefCell<Tensor>>,
}

impl LearningTensor {
    pub fn new(tensor: Rc<RefCell<Tensor>>, gradient: Rc<RefCell<Tensor>>) -> Self {
        Self { tensor, gradient }
    }
    pub fn tensor(&self) -> &Rc<RefCell<Tensor>> {
        &self.tensor
    }
    pub fn gradient(&self) -> &Rc<RefCell<Tensor>> {
        &self.gradient
    }

    /// Back-propagation
    pub fn backward(
        &self,
        error_working_memory: &mut DeltaWorkingMemory,
        device: &Device,
        tape: &Rc<RefCell<Tape>>,
    ) -> Result<Vec<LearningTensor>, Error> {
        let tape: &Tape = &tape.deref().borrow();
        let records: &Vec<Record> = &tape.records();

        for record in records.iter().rev() {
            let operator: &Box<dyn OperatorTrait> = &record.operator().deref().borrow();
            let inputs = record.inputs();
            let output = record.output();

            // Store enabled gradients to optimize them later.
            operator.backward(device, error_working_memory, inputs, output)?;

            // Clip the backward gradients.
            for input in inputs {
                let back_propagated_delta: &mut Tensor = &mut input.gradient().deref().borrow_mut();
                let back_propagated_gradient = device.tensor(
                    back_propagated_delta.rows(),
                    back_propagated_delta.cols(),
                    back_propagated_delta.get_values()?,
                );
                back_propagated_gradient.clip(-1.0, 1.0, back_propagated_delta)?;
            }
        }
        Ok(device.tensors_with_requires_grad())
    }
}
