use std::{marker::PhantomData, ops::Deref};

use crate::{
    gradient_instruction, tensor::Error, tensor::Tensor, BinaryOperator, Category, Device,
    Instruction, OpCode, OptimizerTrait, TensorWithGrad, UnaryModel,
};

use super::assign_streams::assign_streams;

pub struct NeuralMachine<T> {
    device: Device,
    example_input: TensorWithGrad,
    example_output: TensorWithGrad,
    machine_output: TensorWithGrad,
    loss: TensorWithGrad,
    instructions: Vec<Instruction>,
    phantom_data: PhantomData<T>,
}

impl<T> NeuralMachine<T> {
    pub fn try_new(
        device: &Device,
        model: &Box<dyn UnaryModel>,
        loss_operator: &Box<dyn BinaryOperator>,
        _clipped_gradient_norm: f32, // Usually 1.0, so it is not used.
        optimizer: &Box<dyn OptimizerTrait>,
    ) -> Result<Self, Error> {
        // input
        let input_shape = model.input_size();
        let input_len = input_shape[0] * input_shape[1];
        let example_input = device.tensor_with_grad(
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
        let example_output = device.tensor_with_grad(
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

        let tensors = device.tensors_to_optimize().deref().borrow();
        let mut optimizer_instructions = optimizer.optimize(device, &tensors)?;
        instructions.append(&mut optimizer_instructions);

        let machine = NeuralMachine::<T> {
            device: device.clone(),
            example_input,
            example_output,
            machine_output,
            loss,
            instructions,
            phantom_data: Default::default(),
        };

        machine.print();

        machine.assign_streams();

        Ok(machine)
    }

    pub fn loss(&self, expected_output: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
        // Copy expected output
        {
            let example_output: &mut Tensor =
                &mut self.example_output.tensor().deref().borrow_mut();
            let expected_output: &Tensor = &expected_output.tensor().deref().borrow_mut();
            Tensor::copy(expected_output, example_output)?;
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
                let opcode: String = instruction.opcode().clone().into();
                println!("----------------------------------");
                println!(
                    "Debugging instruction {} {} with {} inputs and {} outputs",
                    i,
                    opcode,
                    instruction.inputs().len(),
                    instruction.outputs().len(),
                );
            }

            #[cfg(debug_assertions)]
            for input in instruction.inputs().deref() {
                let opcode: String = instruction.opcode().clone().into();
                assert_eq!(
                    input.is_nan()?,
                    false,
                    "instruction {} {} read nan input {} {}",
                    i,
                    opcode,
                    input.name(),
                    input,
                );
                assert_eq!(
                    input.is_infinite()?,
                    false,
                    "instruction {} {} read inf input {} {}",
                    i,
                    opcode,
                    input.name(),
                    input,
                );
            }

            if debug {
                println!("BEFORE FORWARD");
                self.print_instruction(i, instruction);
                self.print_instruction_inputs_outputs(instruction);
            }

            instruction.execute()?;

            #[cfg(debug_assertions)]
            for output in instruction.outputs().deref() {
                let opcode: String = instruction.opcode().clone().into();
                assert_eq!(
                    output.is_nan()?,
                    false,
                    "instruction {} {} wrote nan output {} {}",
                    i,
                    opcode,
                    output.name(),
                    output,
                );
                assert_eq!(
                    output.is_infinite()?,
                    false,
                    "instruction {} {} wrote inf output {} {}",
                    i,
                    opcode,
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

    pub fn infer(&self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
        // Copy input
        {
            let example_input: &mut Tensor = &mut self.example_input.tensor().deref().borrow_mut();
            let input: &Tensor = &input.tensor().deref().borrow_mut();
            Tensor::copy(input, example_input)?;
        }

        self.forward(Category::Inference)?;

        Ok(self.machine_output.clone())
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
        println!(
            "Inference Instructions: {}",
            self.instructions
                .iter()
                .filter(|i| i.category() == Category::Inference)
                .count()
        );
        println!(
            "Loss Instructions: {}",
            self.instructions
                .iter()
                .filter(|i| i.category() == Category::Loss)
                .count()
        );
        println!(
            "Gradient Instructions: {}",
            self.instructions
                .iter()
                .filter(|i| i.category() == Category::Gradient)
                .count()
        );
        println!(
            "Optimization Instructions: {}",
            self.instructions
                .iter()
                .filter(|i| i.category() == Category::Optimization)
                .count()
        );
        println!("------------------------------");
        for (i, instruction) in self.instructions.iter().enumerate() {
            self.print_instruction(i, instruction);
        }
        println!("------------------------------");
    }

    fn tensor_name(name: usize) -> String {
        "t".to_owned() + name.to_string().as_str()
    }

    fn print_instruction(&self, i: usize, instruction: &Instruction) {
        let opcode: String = instruction.opcode().clone().into();
        let inputs = instruction
            .inputs()
            .iter()
            .map(|x| x.name())
            .map(Self::tensor_name)
            .collect::<Vec<_>>()
            .join(" ");
        let outputs = instruction
            .outputs()
            .iter()
            .map(|x| x.name())
            .map(Self::tensor_name)
            .collect::<Vec<_>>()
            .join(" ");
        let category: String = instruction.category().into();
        println!(
            "{}: INSTRUCTION    {}    {}    {}    // category={}",
            i, opcode, inputs, outputs, category,
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

    fn assign_streams(&self) {
        let machine_inputs = vec![self.example_input.tensor().deref().borrow().name()];
        let instructions = self
            .instructions
            .iter()
            .map(|instruction| {
                let inputs = instruction
                    .inputs()
                    .iter()
                    .map(|x| x.name())
                    .collect::<Vec<_>>();
                let outputs = instruction
                    .outputs()
                    .iter()
                    .map(|x| x.name())
                    .collect::<Vec<_>>();
                (inputs, outputs)
            })
            .collect::<Vec<_>>();
        assign_streams(&machine_inputs, &instructions);
    }
}
