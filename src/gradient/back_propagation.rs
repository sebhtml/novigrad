use std::{cell::RefCell, ops::Deref, rc::Rc};

use crate::{DeltaWorkingMemory, Device, Error, LearningTensor, Record, Tape, Tensor};

use crate::gradient::OperatorTrait;

/// Back-propagation
pub fn back_propagation(
    error_working_memory: &mut DeltaWorkingMemory,
    device: &Device,
    tape: &Rc<RefCell<Tape>>,
) -> Result<Vec<LearningTensor>, Error> {
    let mut enabled_gradients = vec![];
    let tape: &Tape = &tape.deref().borrow();
    let records: &Vec<Record> = &tape.records();

    for record in records.iter().rev() {
        let operator: &Box<dyn OperatorTrait> = &record.operator().deref().borrow();
        let inputs = record.inputs();
        let output = record.output();

        // Store enabled gradients to optimize them later.
        let operator_gradients = operator.backward(device, error_working_memory, inputs, output)?;
        enabled_gradients.extend_from_slice(&operator_gradients);

        // Clip the backward gradients.
        for input in inputs {
            let back_propagated_delta: &mut Tensor = &mut input.gradient().deref().borrow_mut();
            let back_propagated_gradient = device.tensor(
                back_propagated_delta.rows(),
                back_propagated_delta.cols(),
                back_propagated_delta.get_values(),
            );
            back_propagated_gradient.clip(-1.0, 1.0, back_propagated_delta);
        }
    }
    Ok(enabled_gradients)
}
