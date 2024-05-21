use more_asserts::debug_assert_lt;
use std::ops::Deref;

use crate::{
    gradient_instruction, BinaryOperator, Category, Device, Error, Instruction, LossOperator,
    OpCode, Operator, OptimizerTrait, Tensor, TensorF32, UnaryModel,
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
            for instruction in tensor.gradient_instructions().into_iter() {
                let outputs: Vec<TensorF32> =
                    instruction.outputs().deref().clone().into_iter().collect();
                let outputs: Vec<&TensorF32> = outputs.iter().collect();
                let clip_instruction =
                    gradient_instruction!(OpCode::ClipNorm(clipped_gradient_norm), &[], &outputs,);

                instructions.push(instruction.clone());
                instructions.push(clip_instruction);
            }
        }

        let tensors = device.tensors_to_optimize().deref().borrow();
        let mut optimizer_instructions = optimizer.optimize(device, &tensors)?;
        instructions.append(&mut optimizer_instructions);

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

    pub fn loss(&self, expected_output: &Tensor) -> Result<Tensor, Error> {
        // Copy expected output
        {
            let example_output: &mut TensorF32 =
                &mut self.example_output.tensor().deref().borrow_mut();
            let expected_output: &TensorF32 = &expected_output.tensor().deref().borrow_mut();
            TensorF32::copy(expected_output, example_output)?;
        }

        self.forward(Category::Loss)?;

        Ok(self.loss.clone())
    }

    pub fn compute_gradient(&self) -> Result<(), Error> {
        self.forward(Category::Gradient)?;

        Ok(())
    }

    pub fn optimize(&self) -> Result<(), Error> {
        self.forward(Category::Optimization)?;

        Ok(())
    }

    fn forward(&self, category: Category) -> Result<(), Error> {
        let debug = false;
        if debug {
            println!("Debugger for NeuralMachine forward pass");
        }

        // Forward tensors
        #[allow(unused_variables)]
        for (i, instruction) in self
            .instructions
            .iter()
            .enumerate()
            .filter(|(_, i)| i.category() == category)
        {
            if debug {
                println!("----------------------------------");
                println!(
                    "Debugging instruction {} {} with {} inputs and {} outputs",
                    i,
                    instruction.opcode().name(),
                    instruction.inputs().len(),
                    instruction.outputs().len(),
                );
            }

            #[cfg(debug_assertions)]
            for input in instruction.inputs().deref() {
                debug_assert_eq!(
                    input.is_nan()?,
                    false,
                    "instruction {} {} read nan input {} {}",
                    i,
                    instruction.opcode().name(),
                    input.name(),
                    input,
                );
            }

            if debug {
                println!("BEFORE FORWARD");
                self.print_instruction(i, instruction);
                self.print_instruction_inputs_outputs(instruction);
            }

            instruction.forward()?;

            #[cfg(debug_assertions)]
            for output in instruction.outputs().deref() {
                debug_assert_eq!(
                    output.is_nan()?,
                    false,
                    "instruction {} {} wrote nan output {} {}",
                    i,
                    instruction.opcode().name(),
                    output.name(),
                    output,
                );
            }

            if debug {
                println!("AFTER FORWARD");
                let maybe_corrupted_instruction = 81;
                println!(
                    "After forward for instruction {} : instruction {}, inputs: {}",
                    i,
                    maybe_corrupted_instruction,
                    self.instructions[maybe_corrupted_instruction]
                        .inputs()
                        .len()
                );
                self.print_instruction(i, instruction);
                self.print_instruction_inputs_outputs(instruction);
            }
        }

        Ok(())
    }

    pub fn infer(&self, input: &Tensor) -> Result<Tensor, Error> {
        // Copy input
        {
            let example_input: &mut TensorF32 =
                &mut self.example_input.tensor().deref().borrow_mut();
            let input: &TensorF32 = &input.tensor().deref().borrow_mut();
            TensorF32::copy(input, example_input)?;
        }

        self.forward(Category::Inference)?;

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
            self.print_instruction(i, instruction);
        }
        println!("------------------------------");
    }

    fn print_instruction(&self, i: usize, instruction: &Instruction) {
        let opcode = instruction.opcode().name();
        debug_assert_lt!(instruction.inputs().len(), 10);
        let inputs = instruction
            .inputs()
            .iter()
            .map(|x| x.name())
            .collect::<Vec<_>>()
            .join(" ");
        let outputs = instruction
            .outputs()
            .iter()
            .map(|x| x.name())
            .collect::<Vec<_>>()
            .join(" ");
        println!(
            "{}: INSTRUCTION    {}    {}    {}",
            i, opcode, inputs, outputs,
        );
        #[cfg(debug_assertions)]
        println!(
            "Source code location: {} {} {}",
            instruction.file(),
            instruction.line(),
            instruction.column(),
        );
    }

    fn print_instruction_inputs_outputs(&self, instruction: &Instruction) {
        println!("inputs: {}", instruction.inputs().deref().len());

        for (j, input) in instruction.inputs().deref().iter().enumerate() {
            println!("input {}: {}", j, input);
        }

        println!("outputs: {}", instruction.outputs().deref().len());

        for (j, output) in instruction.outputs().deref().iter().enumerate() {
            println!("output {}: {}", j, output);
        }
    }
}
