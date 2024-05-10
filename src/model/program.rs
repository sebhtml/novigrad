use std::{ops::Deref, rc::Rc};

use crate::{Device, Error, Identity, Model, OperatorTrait, Tensor, TensorF32};

pub struct Program {
    example_input: Tensor,
    example_output: Tensor,
    program_output: Tensor,
    loss: Tensor,
    forward_instructions: Vec<Tensor>,
    backward_instructions: Vec<Tensor>,
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
        let example_input = device.tensor(
            Rc::new(Identity::new(device)),
            &vec![],
            input_shape[0],
            input_shape[1],
            vec![0.7; input_len],
            false,
            false,
        );
        // output
        let output_shape = model.output_shape();
        let output_len = output_shape[0] * output_shape[1];
        let example_output = device.tensor(
            Rc::new(Identity::new(device)),
            &vec![],
            output_shape[0],
            output_shape[1],
            vec![0.7; output_len],
            false,
            false,
        );

        let forward_inputs = vec![&example_input];
        let program_output = model.forward(&forward_inputs)?;
        let forward_instructions = program_output.get_tape();

        let loss_inputs = vec![&example_output, &program_output];
        let loss = loss_operator.forward(&loss_inputs)?;
        let backward_instructions = loss.get_tape().clone().into_iter().rev().collect();

        let program = Program {
            example_input,
            example_output,
            program_output,
            loss,
            forward_instructions,
            backward_instructions,
        };
        Ok(program)
    }

    pub fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, Error> {
        // Copy input
        {
            let example_input: &mut TensorF32 =
                &mut self.example_input.tensor().deref().borrow_mut();
            let input: &TensorF32 = &inputs[0].tensor().deref().borrow_mut();
            TensorF32::copy(input, example_input)?;
        }
        // Clear states
        for tensor in self.forward_instructions.iter() {
            tensor.tensor().deref().borrow_mut().zero()?;
            tensor.gradient().deref().borrow_mut().zero()?;
            //tensor.realize()?;
        }
        Ok(self.program_output.clone())
    }
}
