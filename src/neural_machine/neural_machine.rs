use std::{marker::PhantomData, ops::Deref};

use crate::{
    gradient_instruction, tensor::Error, tensor::Tensor, BinaryOperator, Category, Device,
    Instruction, OpCode, OptimizerTrait, TensorWithGrad, UnaryModel,
};

use super::streams::{execute_streams, make_streams, reset_streams, Stream};

pub struct NeuralMachine<T> {
    device: Device,
    example_input: TensorWithGrad,
    example_output: TensorWithGrad,
    machine_output: TensorWithGrad,
    loss: TensorWithGrad,
    inference_instructions: Vec<Instruction>,
    inference_streams: Vec<Stream>,
    loss_instructions: Vec<Instruction>,
    loss_streams: Vec<Stream>,
    gradient_instructions: Vec<Instruction>,
    gradient_streams: Vec<Stream>,
    optimization_instructions: Vec<Instruction>,
    optimization_streams: Vec<Stream>,
    phantom_data: PhantomData<T>,
    max_concurrent_streams: usize,
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
        let mut all_instructions = vec![];

        for tensor in tape.iter() {
            for instruction in tensor.forward_instructions().into_iter() {
                all_instructions.push(instruction);
            }
        }

        for tensor in tape.iter().rev() {
            for instruction in tensor.gradient_instructions().into_iter() {
                let outputs: Vec<Tensor> =
                    instruction.outputs().deref().clone().into_iter().collect();
                let outputs: Vec<&Tensor> = outputs.iter().collect();

                all_instructions.push(instruction);

                for output in outputs {
                    all_instructions.push(gradient_instruction!(
                        OpCode::ClipNorm,
                        &[output],
                        &[output],
                    ));
                }
            }
        }

        let tensors = device.tensors_to_optimize().deref().borrow();
        let mut optimizer_instructions = optimizer.optimize(device, &tensors)?;
        all_instructions.append(&mut optimizer_instructions);

        let inference_instructions = all_instructions
            .clone()
            .into_iter()
            .filter(|i| i.category() == Category::Inference)
            .collect();
        let inference_streams = Self::assign_streams(&example_input, &inference_instructions);
        let loss_instructions = all_instructions
            .clone()
            .into_iter()
            .filter(|i| i.category() == Category::Loss)
            .collect();
        let loss_streams = Self::assign_streams(&example_input, &loss_instructions);
        let gradient_instructions = all_instructions
            .clone()
            .into_iter()
            .filter(|i| i.category() == Category::Gradient)
            .collect();
        let gradient_streams = Self::assign_streams(&example_input, &gradient_instructions);
        let optimization_instructions = all_instructions
            .clone()
            .into_iter()
            .filter(|i| i.category() == Category::Optimization)
            .collect();
        let optimization_streams = Self::assign_streams(&example_input, &optimization_instructions);

        let machine = NeuralMachine::<T> {
            device: device.clone(),
            example_input,
            example_output,
            machine_output,
            loss,
            inference_instructions,
            inference_streams,
            loss_instructions,
            loss_streams,
            gradient_instructions,
            gradient_streams,
            optimization_instructions,
            optimization_streams,
            max_concurrent_streams: 16,
            phantom_data: Default::default(),
        };

        machine.print();

        Ok(machine)
    }

    pub fn loss(&mut self, expected_output: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
        // Copy expected output
        {
            let example_output: &mut Tensor =
                &mut self.example_output.tensor().deref().borrow_mut();
            let expected_output: &Tensor = &expected_output.tensor().deref().borrow_mut();
            Tensor::copy(expected_output, example_output)?;
        }

        self.forward(&Category::Loss)?;

        Ok(self.loss.clone())
    }

    pub fn compute_gradient(&mut self) -> Result<(), Error> {
        self.forward(&Category::Gradient)
    }

    pub fn optimize(&mut self) -> Result<(), Error> {
        self.forward(&Category::Optimization)
    }

    fn forward_with_streams(&mut self, category: &Category) -> Result<(), Error> {
        let streams = match category {
            Category::Inference => &mut self.inference_streams,
            Category::Loss => &mut self.loss_streams,
            Category::Gradient => &mut self.gradient_streams,
            Category::Optimization => &mut self.optimization_streams,
        };
        execute_streams(streams, self.max_concurrent_streams);
        reset_streams(streams);
        Ok(())
    }

    fn forward(&mut self, category: &Category) -> Result<(), Error> {
        self.forward_with_streams(category)?;
        let instructions = match category {
            Category::Inference => &self.inference_instructions,
            Category::Loss => &self.loss_instructions,
            Category::Gradient => &self.gradient_instructions,
            Category::Optimization => &self.optimization_instructions,
        };
        self.forward_without_streams(instructions)
    }

    fn forward_without_streams(&self, instructions: &Vec<Instruction>) -> Result<(), Error> {
        let debug = false;
        if debug {
            println!("Debugger for NeuralMachine forward pass");
        }

        // Forward tensors
        #[allow(unused_variables)]
        for (i, instruction) in instructions.iter().enumerate() {
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
        }

        Ok(())
    }

    pub fn infer(&mut self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
        // Copy input
        {
            let example_input: &mut Tensor = &mut self.example_input.tensor().deref().borrow_mut();
            let input: &Tensor = &input.tensor().deref().borrow_mut();
            Tensor::copy(input, example_input)?;
        }

        self.forward(&Category::Inference)?;

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

        let total_instructions = self.inference_instructions.len()
            + self.loss_instructions.len()
            + self.gradient_instructions.len()
            + self.optimization_instructions.len();
        println!("Instructions: {}", total_instructions);
        println!(
            "Inference Instructions: {}",
            self.inference_instructions.len()
        );
        println!("Loss Instructions: {}", self.loss_instructions.len());
        println!(
            "Gradient Instructions: {}",
            self.gradient_instructions.len()
        );
        println!(
            "Optimization Instructions: {}",
            self.optimization_instructions.len()
        );
        println!("------------------------------");
        for (i, instruction) in self.inference_instructions.iter().enumerate() {
            self.print_instruction(i, instruction);
        }
        println!("------------------------------");
        for (i, instruction) in self.loss_instructions.iter().enumerate() {
            self.print_instruction(i, instruction);
        }
        println!("------------------------------");
        for (i, instruction) in self.gradient_instructions.iter().enumerate() {
            self.print_instruction(i, instruction);
        }
        println!("------------------------------");
        for (i, instruction) in self.optimization_instructions.iter().enumerate() {
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

    fn assign_streams(
        example_input: &TensorWithGrad,
        instructions: &Vec<Instruction>,
    ) -> Vec<Stream> {
        let machine_inputs = vec![example_input.tensor().deref().borrow().name()];
        let instructions = instructions
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
        let streams = make_streams(&machine_inputs, &instructions);
        streams
    }
}
