use std::{cell::RefCell, ops::Deref, rc::Rc};

use crate::{DeltaWorkingMemory, Device, Error, Gradient, Record, Tape, TrainWorkingMemory};

use crate::gradient::OperatorTrait;

/// Back-propagation
pub fn back_propagation(
    working_memory: &mut TrainWorkingMemory,
    error_working_memory: &mut DeltaWorkingMemory,
    device: &Device,
    tape: &Rc<RefCell<Tape>>,
) -> Result<Vec<Gradient>, Error> {
    let mut gradients = vec![];
    let layer_delta = &mut working_memory.layer_delta;
    let tape: &Tape = &tape.deref().borrow();
    let records: &Vec<Record> = &tape.records();
    let layers_count = { tape.records().len() };

    let back_propagated_delta = &mut working_memory.back_propagated_delta;
    back_propagated_delta.reset(0, 0, Default::default());

    for layer_index in (0..layers_count).into_iter().rev() {
        let record = &records[layer_index];
        let inputs = record.inputs();
        let output = record.output();
        let operator: &Box<dyn OperatorTrait> = &record.operator().deref().borrow();

        let (back_propagated_gradient, operator_gradients) = operator.backward(
            device,
            error_working_memory,
            inputs,
            output,
            back_propagated_delta,
        )?;

        back_propagated_gradient
            .deref()
            .borrow_mut()
            .clip(-1.0, 1.0, back_propagated_delta);

        if operator_gradients.len() > 0 {
            debug_assert_eq!(layer_delta.shape(), output.deref().borrow().shape());
        }
        for gradient in operator_gradients {
            gradients.push(gradient);
        }
    }
    Ok(gradients)
}
