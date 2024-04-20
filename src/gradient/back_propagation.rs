use std::mem::swap;
use std::{cell::RefCell, ops::Deref, rc::Rc};

use crate::{Accelerator, DeltaWorkingMemory, Error, Gradient, Tape, Tensor, TrainWorkingMemory};

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
        let tape: &Tape = &tape.deref().borrow();
        let inputs: &Vec<Rc<Tensor>> = &tape.records[layer_index].inputs;
        let output: &Rc<Tensor> = &tape.records[layer_index].output;

        let is_last_layer = layer_index == layers_count - 1;

        {
            let next_layer: Option<Rc<RefCell<Box<dyn OperatorTrait>>>> = if is_last_layer {
                None
            } else {
                let next_layer_index = layer_index + 1;
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
                    let inputs: &Vec<Rc<Tensor>> = { &tape.records[layer_index + 1].inputs };
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

            let layer: &Box<dyn OperatorTrait> =
                &tape.records[layer_index].operator.deref().borrow();
            layer.get_layer_output_delta(
                accelerator,
                error_working_memory,
                &inputs,
                &output,
                back_propagated_delta,
                tmp,
            );

            tmp.clip(-1.0, 1.0, layer_delta)
        }

        {
            let layer: &mut Box<dyn OperatorTrait> =
                &mut tape.records[layer_index].operator.deref().borrow_mut();
            let mut operator_gradients =
                layer.compute_gradients(accelerator, &inputs, layer_delta)?;
            gradients.append(&mut operator_gradients);
        }

        swap(next_layer_delta, layer_delta);
    }
    Ok(gradients)
}
