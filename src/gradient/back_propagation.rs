use std::{cell::RefCell, ops::Deref, rc::Rc};

use crate::{
    Accelerator, DeltaWorkingMemory, Error, Gradient, Record, Tape, Tensor, TrainWorkingMemory,
};

use crate::gradient::OperatorTrait;

/// Back-propagation
pub fn back_propagation(
    working_memory: &mut TrainWorkingMemory,
    error_working_memory: &mut DeltaWorkingMemory,
    accelerator: &Accelerator,
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

        let (back_propagated_gradient, operator_gradients) = backward(
            operator,
            accelerator,
            error_working_memory,
            inputs,
            output,
            back_propagated_delta,
            layer_delta,
        )?;

        back_propagated_gradient.clip(-1.0, 1.0, back_propagated_delta);

        if operator_gradients.len() > 0 {
            debug_assert_eq!(layer_delta.shape(), output.shape());
        }
        for gradient in operator_gradients {
            gradients.push(gradient);
        }
    }
    Ok(gradients)
}

fn backward(
    operator: &Box<dyn OperatorTrait>,
    accelerator: &Accelerator,
    error_working_memory: &mut DeltaWorkingMemory,
    inputs: &Vec<Rc<Tensor>>,
    output: &Rc<Tensor>,
    back_propagated_delta: &mut Tensor,
    layer_delta: &mut Tensor,
) -> Result<(Tensor, Vec<Gradient>), Error> {
    operator.get_layer_output_delta(
        accelerator,
        error_working_memory,
        inputs,
        output,
        back_propagated_delta,
        layer_delta,
    );

    let operator_gradients = operator.compute_gradients(accelerator, inputs, layer_delta)?;
    operator.backward2(inputs, accelerator, layer_delta, back_propagated_delta);

    Ok((back_propagated_delta.clone(), operator_gradients))
}
