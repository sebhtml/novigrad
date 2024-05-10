use std::rc::Rc;

use crate::{Device, Error, Identity, Model, OperatorTrait, Tensor};

pub struct Program {
    input: Tensor,
    output: Tensor,
    program_output: Tensor,
    loss: Tensor,
    instructions: Vec<Tensor>,
}

impl Program {
    pub fn try_new(
        device: &Device,
        model: &Box<dyn Model>,
        loss_operator: &Box<dyn OperatorTrait>,
    ) -> Result<Self, Error> {
        // input
        let input_shape = model.input_shape();
        let input_len = input_shape[0] * input_shape[1];
        let input = device.tensor(
            Rc::new(Identity::new(device)),
            &vec![],
            input_shape[0],
            input_shape[1],
            vec![0.0; input_len],
            false,
            false,
        );
        // output
        let output_shape = model.output_shape();
        let output_len = output_shape[0] * output_shape[1];
        let output = device.tensor(
            Rc::new(Identity::new(device)),
            &vec![],
            output_shape[0],
            output_shape[1],
            vec![0.0; output_len],
            false,
            false,
        );

        let inputs = vec![&input];
        let program_output = model.forward(&inputs)?;
        let loss = loss_operator.forward(&inputs)?;
        let instructions = loss.get_tape();
        let program = Program {
            input,
            output,
            program_output,
            loss,
            instructions,
        };
        Ok(program)
    }

    pub fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, Error> {
        // TODO
        // assign input
        // assign output
        // call realize on loss.
        // return program_output
        Ok(self.program_output.clone())
    }
}
