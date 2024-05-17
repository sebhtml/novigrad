use std::{ops::Deref, rc::Rc};

use crate::{
    BinaryOperator, Clip, Device, Error, IdentityBackward, Instruction, LossOperator,
    OptimizerTrait, Tensor, TensorF32, UnaryModel,
};

pub struct NeuralMachine {
    device: Device,
    example_input: Tensor,
    example_output: Tensor,
    program_output: Tensor,
    loss: Tensor,
    instructions: Vec<Instruction>,
}

impl NeuralMachine {
    pub fn try_new(
        device: &Device,
        model: &Box<dyn UnaryModel>,
        loss_operator: &Box<dyn LossOperator>,
        clipped_gradient_norm: f32,
        optimizer: &Box<dyn OptimizerTrait>,
    ) -> Result<Self, Error> {
        // input
        let input_shape = model.input_size();
        let input_len = input_shape[0] * input_shape[1];
        let example_input = device.tensor(
            input_shape[0],
            input_shape[1],
            vec![0.7; input_len],
            &[],
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
            &[],
            false,
            false,
        );

        let program_output = model.forward(&example_input)?;
        let loss =
            BinaryOperator::forward(loss_operator.deref(), &example_output, &program_output)?;
        let tape = loss.get_tape();
        let mut instructions = vec![];

        for tensor in tape.iter() {
            for instruction in tensor.forward_instructions().into_iter() {
                instructions.push(instruction);
            }
        }

        for tensor in tape.iter().rev() {
            for instruction in tensor.backward_instructions().into_iter() {
                let outputs: Vec<TensorF32> =
                    instruction.outputs().deref().clone().into_iter().collect();
                let outputs: Vec<&TensorF32> = outputs.iter().collect();
                let clip_instruction = Instruction::new(
                    Rc::new(Clip::new(clipped_gradient_norm)),
                    &[],
                    &outputs,
                    instruction.gradient_pathway(),
                );

                instructions.push(instruction.clone());
                instructions.push(clip_instruction);
            }
        }

        let mut instructions = Self::optimize_softmax_and_cross_entropy_loss(device, &instructions);

        if false
        // TODO
        {
            let tensors = device.tensors_to_optimize().deref().borrow();
            let mut optimizer_instructions = optimizer.optimize(device, &tensors)?;
            instructions.append(&mut optimizer_instructions);
        }

        let program = NeuralMachine {
            device: device.clone(),
            example_input,
            example_output,
            program_output,
            loss,
            instructions,
        };

        program.print();
        Ok(program)
    }

    pub fn loss(&self) -> Result<Tensor, Error> {
        Ok(self.loss.clone())
    }
}

impl NeuralMachine {
    pub fn forward(&self, input: &Tensor, expected_output: &Tensor) -> Result<Tensor, Error> {
        //println!("NeuralMachine forward");
        // Copy input
        {
            let example_input: &mut TensorF32 =
                &mut self.example_input.tensor().deref().borrow_mut();
            let input: &TensorF32 = &input.tensor().deref().borrow_mut();
            TensorF32::copy(input, example_input)?;
        }
        // Copy expected output
        {
            let example_output: &mut TensorF32 =
                &mut self.example_output.tensor().deref().borrow_mut();
            let expected_output: &TensorF32 = &expected_output.tensor().deref().borrow_mut();
            TensorF32::copy(expected_output, example_output)?;
        }
        // Forward tensors
        #[allow(unused_variables)]
        for (i, instruction) in self.instructions.iter().enumerate() {
            //cprintln!("Forward instruction {} {}", i, instruction.operator().name(),);
            #[cfg(debug_assertions)]
            for input in instruction.inputs().deref() {
                debug_assert_eq!(
                    input.is_nan()?,
                    false,
                    "instruction {} {} read nan input {} {}",
                    i,
                    instruction.operator().name(),
                    input.name(),
                    input,
                );
            }

            instruction.forward()?;

            #[cfg(debug_assertions)]
            for output in instruction.outputs().deref() {
                debug_assert_eq!(
                    output.is_nan()?,
                    false,
                    "instruction {} {} wrote nan output {} {}",
                    i,
                    instruction.operator().name(),
                    output.name(),
                    output,
                );
            }

            // TODO impl Display
            /*
            println!(
                "{} -> {}, {} inputs, {} outputs",
                i,
                instruction.operator().name(),
                instruction.inputs().len(),
                instruction.outputs().len()
            );

            println!("inputs:");

            for inputs in instruction.inputs().deref().iter() {
                println!("inputs {}", inputs);
            }

            println!("outputs:");

            for output in instruction.outputs().deref().iter() {
                println!("output {}", output);
            }
             */
        }
        Ok(self.program_output.clone())
    }

    pub fn print(&self) {
        println!("------------------------------");
        println!("Booting Neural Machine...");
        println!("Neural program compiled with Novigrad");

        println!("Tensors: {}", self.device.tensor_count());
        println!("Parameters: {}", self.device.parameter_count());

        let input_size: Vec<usize> = self
            .example_input
            .tensor()
            .deref()
            .borrow()
            .size()
            .deref()
            .borrow()
            .clone();
        println!(
            "Input size: [{}]",
            input_size
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );

        let output_size: Vec<usize> = self
            .example_output
            .tensor()
            .deref()
            .borrow()
            .size()
            .deref()
            .borrow()
            .clone();
        println!(
            "Output size: [{}]",
            output_size
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );

        println!("Instructions: {}", self.instructions.len());
        println!("------------------------------");
        for (i, instruction) in self.instructions.iter().enumerate() {
            println!(
                "{}: INSTRUCTION    {}    {}    {}",
                i,
                instruction.operator().name(),
                instruction
                    .inputs()
                    .iter()
                    .map(|x| x.name())
                    .collect::<Vec<_>>()
                    .join(" "),
                instruction
                    .outputs()
                    .iter()
                    .map(|x| x.name())
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
        println!("------------------------------");
    }

    pub fn optimize_softmax_and_cross_entropy_loss(
        _device: &Device,
        instructions: &Vec<Instruction>,
    ) -> Vec<Instruction> {
        let mut new_instructions = vec![];
        let mut i = 0;
        while i < instructions.len() {
            if i + 3 < instructions.len() {
                if instructions[i + 0].operator().name() == "CrossEntropyLossBackward"
                    && instructions[i + 1].operator().name() == "Clip"
                    && instructions[i + 2].operator().name() == "SoftmaxBackward"
                    && instructions[i + 3].operator().name() == "Clip"
                {
                    new_instructions.push(instructions[i + 0].clone());
                    new_instructions.push(instructions[i + 1].clone());
                    let softmax_backward_input_gradient = &instructions[i + 2].inputs().deref()[1];
                    new_instructions.push(Instruction::new(
                        Rc::new(IdentityBackward::default()),
                        &[softmax_backward_input_gradient],
                        &instructions[i + 2].outputs().iter().collect::<Vec<_>>(),
                        instructions[i + 2].gradient_pathway(),
                    ));
                    new_instructions.push(instructions[i + 3].clone());
                    i += 4;
                } else {
                    new_instructions.push(instructions[i].clone());
                    i += 1;
                }
            } else {
                new_instructions.push(instructions[i].clone());
                i += 1;
            }
        }
        new_instructions
    }
}
