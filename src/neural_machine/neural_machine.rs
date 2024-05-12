use std::{cell::RefCell, ops::Deref, rc::Rc};

use crate::{
    BinaryOperator, Device, Error, Instruction, Model, Operator, Tensor, TensorF32, UnaryOperator,
    Zero,
};

use super::instruction;

pub struct NeuralMachine {
    device: Device,
    example_input: Tensor,
    example_output: Tensor,
    program_output: Tensor,
    loss: Tensor,
    forward_instructions: Vec<Instruction>,
    backward_instructions: Vec<Instruction>,
}

impl NeuralMachine {
    pub fn try_new(
        device: &Device,
        model: &(impl UnaryOperator + Model),
        loss_operator: &(impl BinaryOperator + Operator),
    ) -> Result<Self, Error> {
        // input
        let input_shape = model.input_size();
        let input_len = input_shape[0] * input_shape[1];
        let example_input = device.tensor(
            input_shape[0],
            input_shape[1],
            vec![0.7; input_len],
            false,
            false,
        );
        // output
        let output_shape = model.output_size();
        let output_len = output_shape[0] * output_shape[1];
        let example_output = device.tensor(
            output_shape[0],
            output_shape[1],
            vec![0.7; output_len],
            false,
            false,
        );

        let program_output = model.forward(&example_input)?;
        let mut forward_instructions = vec![];

        let tape = program_output.get_tape();

        for tensor in tape {
            let instruction = tensor.forward_instructions().deref().borrow()[0].to_owned();
            let outputs: Vec<Tensor> = instruction.outputs().deref().clone().into_iter().collect();
            let outputs: Vec<&Tensor> = outputs.iter().collect();
            let zero_instruction = Instruction::new(Rc::new(Zero::default()), &[], &outputs);
            forward_instructions.push(zero_instruction);
            forward_instructions.push(instruction);
        }

        let loss = BinaryOperator::forward(loss_operator, &example_output, &program_output)?;
        let backward_instructions = loss
            .get_tape()
            .clone()
            .into_iter()
            .rev()
            .map(|x| x.forward_instructions().deref().borrow()[0].to_owned())
            .collect();

        let program = NeuralMachine {
            device: device.clone(),
            example_input,
            example_output,
            program_output,
            loss,
            forward_instructions,
            backward_instructions,
        };
        Ok(program)
    }

    pub fn loss(&self, expected_output: &Tensor) -> Result<Tensor, Error> {
        // Copy expected output
        {
            let example_output: &mut TensorF32 =
                &mut self.example_output.tensor().deref().borrow_mut();
            let expected_output: &TensorF32 = &expected_output.tensor().deref().borrow_mut();
            TensorF32::copy(expected_output, example_output)?;
        }
        let loss = &self.loss;
        loss.forward_instructions().deref().borrow()[0].forward()?;
        Ok(loss.clone())
    }

    /// Back-propagation
    pub fn backward(&self) -> Result<Rc<RefCell<Vec<Tensor>>>, Error> {
        for instruction in self.backward_instructions.iter() {
            instruction.backward()?;

            let inputs: Vec<_> = instruction.inputs().iter().collect();
            for input in inputs {
                if !input.requires_grad() {
                    continue;
                }
                let input_gradient: &mut TensorF32 = &mut input.gradient().deref().borrow_mut();
                // Clip the backward gradients.
                input_gradient.clip(-1.0, 1.0)?;
            }
        }
        Ok(self.device.tensors_to_optimize().clone())
    }
}

impl UnaryOperator for NeuralMachine {
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        //println!("NeuralMachine forward");
        // Copy input
        {
            let example_input: &mut TensorF32 =
                &mut self.example_input.tensor().deref().borrow_mut();
            let input: &TensorF32 = &input.tensor().deref().borrow_mut();
            TensorF32::copy(input, example_input)?;
        }
        // Forward tensors
        for instruction in self.forward_instructions.iter() {
            instruction.forward()?;
        }
        Ok(self.program_output.clone())
    }
}
