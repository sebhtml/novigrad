use std::ops::Deref;

use crate::{
    gradient_instruction, new_tensor_with_grad,
    tensor::{Error, Tensor},
    BinaryOperator, Device, Instruction, OpCode, OptimizerTrait, TensorWithGrad, UnaryModel,
};

pub struct NeuralProgram {
    pub example_input: TensorWithGrad,
    pub example_output: TensorWithGrad,
    pub machine_output: TensorWithGrad,
    pub loss: TensorWithGrad,
    pub instructions: Vec<Instruction>,
}

impl NeuralProgram {
    pub fn try_new(
        device: &Device,
        model: &Box<dyn UnaryModel>,
        loss_operator: &Box<dyn BinaryOperator>,
        optimizer: &Box<dyn OptimizerTrait>,
    ) -> Result<NeuralProgram, Error> {
        // input
        let input_shape = model.input_size();
        let input_len = input_shape[0] * input_shape[1];
        let example_input = new_tensor_with_grad!(
            device,
            input_shape[0],
            input_shape[1],
            vec![0.7; input_len],
            &[],
            false,
            false,
        )?;
        // output
        let output_shape = model.output_size();
        let output_len = output_shape[0] * output_shape[1];
        let example_output = new_tensor_with_grad!(
            device,
            output_shape[0],
            output_shape[1],
            vec![0.7; output_len],
            &[],
            false,
            false,
        )?;

        let machine_output = model.forward(&example_input)?;
        let loss =
            BinaryOperator::forward(loss_operator.deref(), &example_output, &machine_output)?;
        let tape = loss.get_tape();
        let mut instructions = vec![];

        for tensor in tape.iter() {
            for instruction in tensor.forward_instructions().into_iter() {
                instructions.push(instruction);
            }
        }

        for tensor in tape.iter().rev() {
            for instruction in tensor.gradient_instructions().into_iter() {
                let outputs: Vec<Tensor> =
                    instruction.outputs().deref().clone().into_iter().collect();
                let outputs: Vec<&Tensor> = outputs.iter().collect();

                instructions.push(instruction);

                for output in outputs {
                    instructions.push(gradient_instruction!(
                        OpCode::ClipNorm,
                        &[output],
                        &[output],
                    ));
                }
            }
        }

        let tensors = device.tensors_to_optimize();
        let mut optimizer_instructions = optimizer.optimize(device, &tensors)?;
        instructions.append(&mut optimizer_instructions);

        let program = NeuralProgram {
            example_input,
            example_output,
            machine_output,
            loss,
            instructions,
        };
        Ok(program)
    }
}
