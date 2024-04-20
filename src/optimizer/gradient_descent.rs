use std::{cell::RefCell, ops::Deref, rc::Rc};

use crate::{Accelerator, Gradient, OperatorEnum, OperatorTrait, OptimizerTrait, Tape};

#[derive(Default)]
pub struct GradientDescent {}

impl OptimizerTrait for GradientDescent {
    fn optimize(
        &self,
        tape: &Rc<RefCell<Tape>>,
        gradients: Vec<Gradient>,
        accelerator: &Accelerator,
    ) {
        let layers_count = {
            let tape = tape.deref().borrow();
            tape.records.len()
        };

        let learning_rate: f32 = 0.5;
        for layer_index in 0..layers_count {
            let tape = tape.deref().borrow();
            let layer: &mut OperatorEnum =
                &mut tape.records[layer_index].operator.deref().borrow_mut();
            let op_result = layer.commit_change(accelerator, learning_rate);
            op_result.expect("Ok");
        }
    }
}
