use std::{collections::HashSet, ops::Deref};

use crate::{
    gradient_instruction, instruction, new_tensor, new_tensor_with_grad,
    opcode::OpCode,
    tensor::{Error, Tensor},
    BinaryOperator, Category, Device, Instruction, OperatorAttributes, OptimizerTrait,
    TensorWithGrad, UnaryModel,
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
        model: &impl UnaryModel,
        loss_operator: &impl BinaryOperator,
        optimizer: &impl OptimizerTrait,
        clipped_gradient_norm: bool,
    ) -> Result<NeuralProgram, Error> {
        let zero = new_tensor!(device, 1, 1, vec![0.0])?;
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
        let loss = BinaryOperator::forward(loss_operator, &example_output, &machine_output)?;
        let tape = loss.get_tape();
        let mut instructions = vec![];

        let mut processed_forward_tensors = HashSet::<usize>::new();
        for tensor in tape.iter() {
            let tensor_name = tensor.tensor().name();
            if processed_forward_tensors.contains(&tensor_name) {
                continue;
            }
            for instruction in tensor.forward_instructions().into_iter() {
                instructions.push(instruction);
            }
            processed_forward_tensors.insert(tensor_name);
        }

        // Gradient instructions
        let internal_tensors = device.internal_tensors();
        for tensor in internal_tensors.iter() {
            let inst = instruction!(
                OpCode::ScalarMul,
                OperatorAttributes::None,
                &[&zero, &tensor.gradient()],
                &[&tensor.gradient()],
                Category::Gradient,
            );
            instructions.push(inst);
        }

        let mut processed_backward_tensors = HashSet::<usize>::new();
        for tensor in tape.iter().rev() {
            let tensor_name = tensor.tensor().name();
            if processed_backward_tensors.contains(&tensor_name) {
                continue;
            }
            for instruction in tensor.gradient_instructions().into_iter() {
                let outputs: Vec<Tensor> =
                    instruction.outputs().deref().clone().into_iter().collect();
                let outputs: Vec<&Tensor> = outputs.iter().collect();

                instructions.push(instruction);

                if clipped_gradient_norm {
                    for output in outputs {
                        instructions.push(gradient_instruction!(
                            OpCode::ClipNorm,
                            OperatorAttributes::None,
                            &[output],
                            &[output],
                        ));
                    }
                }
            }
            processed_backward_tensors.insert(tensor_name);
        }

        // Optimization instructions
        let parameter_tensors = device.parameter_tensors();
        let mut optimizer_instructions = optimizer.optimize(device, &parameter_tensors)?;
        instructions.append(&mut optimizer_instructions);

        for tensor in parameter_tensors.iter() {
            let inst = instruction!(
                OpCode::ScalarMul,
                OperatorAttributes::None,
                &[&zero, &tensor.gradient()],
                &[&tensor.gradient()],
                Category::Optimization,
            );
            instructions.push(inst);
        }

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
