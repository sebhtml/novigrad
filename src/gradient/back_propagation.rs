use std::borrow::Borrow;
use std::mem::swap;
use std::{cell::RefCell, ops::Deref, rc::Rc};

use crate::{
    Accelerator, DeltaWorkingMemory, Error, Gradient, OperatorEnum, Tape, Tensor,
    TrainWorkingMemory,
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
    let next_layer_delta = &mut working_memory.next_layer_delta;
    let layer_delta = &mut working_memory.layer_delta;
    let layers_count = {
        let tape = tape.deref().borrow();
        tape.records.len()
    };

    next_layer_delta.assign(accelerator, &Default::default());
    for layer_index in (0..layers_count).into_iter().rev() {
        let layer_output = &mut working_memory.layer_output;
        {
            let tape = tape.deref().borrow();
            let tensor = tape.records[layer_index].output.borrow();
            layer_output.assign(accelerator, tensor);
        }

        let is_last_layer = layer_index == layers_count - 1;

        let inputs: Vec<Tensor> = {
            let tape = tape.deref().borrow();
            tape.records[layer_index].inputs.clone()
        };

        {
            let next_layer: Option<Rc<RefCell<OperatorEnum>>> = if is_last_layer {
                None
            } else {
                let next_layer_index = layer_index + 1;
                let tape = tape.deref().borrow();
                let operator = tape.records[next_layer_index].operator.clone();
                Some(operator)
            };

            let tmp = &mut working_memory.tmp;
            let back_propagated_delta = &mut working_memory.back_propagated_delta;

            match next_layer {
                None => {
                    // use the output of the loss function¸
                    back_propagated_delta.assign(accelerator, next_layer_delta);
                }
                Some(next_layer) => {
                    let inputs: Vec<Tensor> = {
                        let tape = tape.deref().borrow();
                        tape.records[layer_index + 1].inputs.clone()
                    };
                    // Hidden layer
                    let next_layer = next_layer.deref();
                    next_layer.borrow().backward(
                        &inputs,
                        accelerator,
                        next_layer_delta,
                        back_propagated_delta,
                    );
                }
            }

            let tape = tape.deref().borrow();
            let layer: &OperatorEnum = &tape.records[layer_index].operator.deref().borrow();
            layer.get_layer_output_delta(
                accelerator,
                error_working_memory,
                &inputs,
                layer_output,
                back_propagated_delta,
                tmp,
            );

            tmp.clip(-1.0, 1.0, layer_delta)
        }

        {
            let tape = tape.deref().borrow();
            let layer: &mut OperatorEnum =
                &mut tape.records[layer_index].operator.deref().borrow_mut();
            let mut operator_gradients =
                layer.compute_gradients(accelerator, &inputs, layer_delta)?;
            gradients.append(&mut operator_gradients);
        }

        swap(next_layer_delta, layer_delta);
    }
    Ok(gradients)
}
