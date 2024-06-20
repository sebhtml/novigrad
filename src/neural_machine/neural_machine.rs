use std::{marker::PhantomData, ops::Deref, sync::Arc};

use crate::schedulers::SchedulerTrait;
use crate::stream::StreamTrait;
use crate::{
    neural_machine::streams::stream::print_streams, neural_program::NeuralProgram,
    schedulers::StreamExecutor, stream::DeviceStream, tensor::Error, Category, Device, Instruction,
    TensorWithGrad,
};

use super::streams::{
    instruction::make_simple_instructions,
    stream::{make_streams, Stream},
    verify_machine_inputs,
};

pub struct NeuralMachine<T, Scheduler>
where
    Scheduler: SchedulerTrait<StreamExecutor>,
{
    device: Device,
    io_stream: DeviceStream,
    example_input: TensorWithGrad,
    example_output: TensorWithGrad,
    machine_output: TensorWithGrad,
    loss: TensorWithGrad,
    inference_instructions: Arc<Vec<Instruction>>,
    inference_streams: Arc<Vec<Stream>>,
    inference_scheduler: Scheduler,
    loss_instructions: Arc<Vec<Instruction>>,
    loss_streams: Arc<Vec<Stream>>,
    loss_scheduler: Scheduler,
    gradient_instructions: Arc<Vec<Instruction>>,
    gradient_streams: Arc<Vec<Stream>>,
    gradient_scheduler: Scheduler,
    optimization_instructions: Arc<Vec<Instruction>>,
    optimization_streams: Arc<Vec<Stream>>,
    optimization_scheduler: Scheduler,
    phantom_data: PhantomData<T>,
}

impl<T, Scheduler> NeuralMachine<T, Scheduler>
where
    Scheduler: SchedulerTrait<StreamExecutor>,
{
    pub fn try_new(device: &Device, program: NeuralProgram) -> Result<Self, Error> {
        let all_instructions = program.instructions;
        let inference_instructions = all_instructions
            .clone()
            .into_iter()
            .filter(|i| i.category() == Category::Inference)
            .collect();
        let inference_instructions = Arc::new(inference_instructions);

        let loss_instructions = all_instructions
            .clone()
            .into_iter()
            .filter(|i| i.category() == Category::Loss)
            .collect();
        let loss_instructions = Arc::new(loss_instructions);

        let gradient_instructions = all_instructions
            .clone()
            .into_iter()
            .filter(|i| i.category() == Category::Gradient)
            .collect();
        let gradient_instructions = Arc::new(gradient_instructions);

        let optimization_instructions = all_instructions
            .clone()
            .into_iter()
            .filter(|i| i.category() == Category::Optimization)
            .collect();
        let optimization_instructions = Arc::new(optimization_instructions);

        #[cfg(feature = "cuda")]
        // TODO we need CUDA streams exposed by DeviceTrait to bump this to 16.
        let maximum_device_streams = 1;
        #[cfg(not(feature = "cuda"))]
        let maximum_device_streams = 16;

        let example_input = program.example_input;
        let example_output = program.example_output;
        let machine_output = program.machine_output;
        let loss = program.loss;

        let inference_streams = Self::assign_streams(&example_input, &inference_instructions);
        let inference_streams = Arc::new(inference_streams);
        let loss_streams = Self::assign_streams(&example_input, &loss_instructions);
        let loss_streams = Arc::new(loss_streams);
        let gradient_streams = Self::assign_streams(&example_input, &gradient_instructions);
        let gradient_streams = Arc::new(gradient_streams);
        let optimization_streams = Self::assign_streams(&example_input, &optimization_instructions);
        let optimization_streams = Arc::new(optimization_streams);

        let handler = StreamExecutor::new();
        let mut inference_scheduler = Scheduler::new(
            device,
            maximum_device_streams,
            &inference_streams,
            &handler,
            &inference_instructions,
        );
        let mut loss_scheduler = Scheduler::new(
            device,
            maximum_device_streams,
            &loss_streams,
            &handler,
            &loss_instructions,
        );
        let mut gradient_scheduler = Scheduler::new(
            device,
            maximum_device_streams,
            &gradient_streams,
            &handler,
            &gradient_instructions,
        );
        let mut optimization_scheduler = Scheduler::new(
            device,
            maximum_device_streams,
            &optimization_streams,
            &handler,
            &optimization_instructions,
        );
        inference_scheduler.start();
        loss_scheduler.start();
        gradient_scheduler.start();
        optimization_scheduler.start();

        let machine = NeuralMachine::<T, Scheduler> {
            device: device.clone(),
            io_stream: device.new_stream()?,
            example_input,
            example_output,
            machine_output,
            loss,
            inference_instructions,
            inference_streams,
            inference_scheduler,
            loss_instructions,
            loss_streams,
            loss_scheduler,
            gradient_instructions,
            gradient_streams,
            gradient_scheduler,
            optimization_instructions,
            optimization_streams,
            optimization_scheduler,
            phantom_data: Default::default(),
        };

        machine.print();

        Ok(machine)
    }

    pub fn instructions(&self, category: &Category) -> impl Deref<Target = Vec<Instruction>> {
        match category {
            Category::Inference => self.inference_instructions.clone(),
            Category::Loss => self.loss_instructions.clone(),
            Category::Gradient => self.gradient_instructions.clone(),
            Category::Optimization => self.optimization_instructions.clone(),
        }
    }

    pub fn loss(&mut self, expected_output: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
        // Copy expected output
        {
            let example_output = &self.example_output.tensor();
            let expected_output = &expected_output.tensor();
            self.device
                .copy_to(expected_output, example_output, &self.io_stream)?;
            self.io_stream.wait_for()?;
        }

        self.forward(&Category::Loss)?;

        Ok(self.loss.clone())
    }

    pub fn compute_gradient(&mut self) -> Result<(), Error> {
        self.forward(&Category::Gradient)?;
        Ok(())
    }

    pub fn optimize(&mut self) -> Result<(), Error> {
        self.forward(&Category::Optimization)?;
        Ok(())
    }

    fn forward_with_streams(&mut self, category: &Category) -> Result<(), Error> {
        let scheduler = match category {
            Category::Inference => &mut self.inference_scheduler,
            Category::Loss => &mut self.loss_scheduler,
            Category::Gradient => &mut self.gradient_scheduler,
            Category::Optimization => &mut self.optimization_scheduler,
        };
        scheduler.execute();
        Ok(())
    }

    fn forward(&mut self, category: &Category) -> Result<(), Error> {
        self.forward_with_streams(category)?;
        Ok(())
    }

    pub fn infer(&mut self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
        // Copy input
        {
            let example_input = &self.example_input.tensor();
            let input = &input.tensor();
            self.device.copy_to(input, example_input, &self.io_stream)?;
            self.io_stream.wait_for()?;
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

        let input_size: Vec<usize> = self.example_input.tensor().size().clone();
        println!(
            "Input size: [{}]",
            input_size
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );

        let output_size: Vec<usize> = self.example_output.tensor().size().clone();
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

        /*
        println!("Tensors");
        for tensor in self.device.tensors().iter() {
            println!("{}", tensor);
        }
         */

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

        print_streams("inference", &self.inference_streams);
        print_streams("loss", &self.loss_streams);
        print_streams("gradient", &self.gradient_streams);
        print_streams("optimization", &self.optimization_streams);
    }

    fn tensor_name(name: usize) -> String {
        "t".to_owned() + name.to_string().as_str()
    }

    fn print_instruction(&self, i: usize, instruction: &Instruction) {
        let opcode: String = instruction.opcode().into();
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

    fn _print_instruction_inputs_outputs(&self, instruction: &Instruction) {
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
        let machine_inputs = vec![example_input.tensor().name()];
        let simple_instructions = make_simple_instructions(instructions);
        verify_machine_inputs(&machine_inputs, &simple_instructions);
        let minimum_write_before_read_for_new_stream = 4;
        let minimum_dependents_for_stream = 12;
        let minimum_stream_instructions = 32;

        make_streams(
            &simple_instructions,
            minimum_write_before_read_for_new_stream,
            minimum_dependents_for_stream,
            minimum_stream_instructions,
        )
    }
}

impl<T, Scheduler> Drop for NeuralMachine<T, Scheduler>
where
    Scheduler: SchedulerTrait<StreamExecutor>,
{
    fn drop(&mut self) {
        self.inference_scheduler.stop();
        self.loss_scheduler.stop();
        self.gradient_scheduler.stop();
        self.optimization_scheduler.stop();
    }
}
