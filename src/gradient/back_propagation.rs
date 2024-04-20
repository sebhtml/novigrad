use std::mem::swap;
use std::{cell::RefCell, ops::Deref, rc::Rc};

use crate::{Accelerator, DeltaWorkingMemory, Error, Gradient, Record, Tape, TrainWorkingMemory};

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
    let tape: &Tape = &tape.deref().borrow();
    let records: &Vec<Record> = &tape.records();
    let layers_count = { tape.records().len() };
    let tmp = &mut working_memory.tmp;

    let back_propagated_delta = &mut working_memory.back_propagated_delta;
    next_layer_delta.reset(1, 333, 0.0);
    back_propagated_delta.reset(1, 444, 0.0);

    /*
               simple dataset
               Ok shapes
               ----
    Layer 8 next_layer_delta (1, 333)
    After call to backward (1, 444)
    After call to clip (1, 444)
    Layer 7 next_layer_delta (1, 444)
    After call to backward (1, 16)
    After call to clip (1, 16)
    Layer 6 next_layer_delta (1, 16)
    After call to backward (1, 16)
    After call to clip (1, 16)
    Layer 5 next_layer_delta (1, 16)
    After call to backward (1, 32)
    After call to clip (1, 32)
    Layer 4 next_layer_delta (1, 32)
    After call to backward (1, 32)
    After call to clip (1, 32)
    Layer 3 next_layer_delta (1, 32)
    After call to backward (1, 96)
    After call to clip (6, 16)
    Layer 2 next_layer_delta (6, 16)
    After call to backward (6, 16)
    After call to clip (6, 16)
    Layer 1 next_layer_delta (6, 16)
    After call to backward (6, 16)
    After call to clip (6, 16)
    Layer 0 next_layer_delta (6, 16)
    After call to backward (6, 32)
    After call to clip (6, 32)


            */
    println!("----");
    for layer_index in (0..layers_count).into_iter().rev() {
        println!(
            "Layer {} next_layer_delta {:?}",
            layer_index,
            next_layer_delta.shape()
        );
        let record = &records[layer_index];
        let inputs = record.inputs();
        let output = record.output();
        let operator: &Box<dyn OperatorTrait> = &record.operator().deref().borrow();

        if layer_index != layers_count - 1 {
            let next_layer_index = layer_index + 1;
            let next_record = &records[next_layer_index];
            let next_operator: &Box<dyn OperatorTrait> = &next_record.operator().deref().borrow();
            let next_inputs = next_record.inputs();

            next_operator.backward(
                next_inputs,
                accelerator,
                next_layer_delta,
                back_propagated_delta,
            );
        };
        println!("After call to backward {:?}", back_propagated_delta.shape());

        operator.get_layer_output_delta(
            accelerator,
            error_working_memory,
            inputs,
            output,
            back_propagated_delta,
            tmp,
        );
        tmp.clip(-1.0, 1.0, layer_delta);
        println!("After call to clip {:?}", layer_delta.shape());

        let mut operator_gradients =
            operator.compute_gradients(accelerator, inputs, layer_delta)?;

        gradients.append(&mut operator_gradients);

        swap(next_layer_delta, layer_delta);
    }
    Ok(gradients)
}
