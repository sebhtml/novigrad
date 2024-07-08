use std::time::SystemTime;

use crate::{
    batch::make_batches,
    datasets::DatasetDetails,
    display::TensorPrinter,
    neural_program::NeuralProgram,
    schedulers::DefaultStreamScheduler,
    tensor::{Error, Tensor},
    BinaryOperator, Device, NeuralMachine, OptimizerTrait, TensorWithGrad, UnaryModel,
};

fn print_device_mem_info(device: &Device) -> Result<(), Error> {
    let mem_info = &device.get_memory_info()?;
    println!(
        "Device memory  used: {}, free: {}, total: {}",
        mem_info.used, mem_info.free, mem_info.total,
    );
    Ok(())
}

#[derive(Clone)]
pub struct Metrics {
    pub total_loss: f32,
}

pub struct NeuralMachineTestOutput {
    pub initial_metrics: Metrics,
    pub final_metrics: Metrics,
    pub expected_argmax_values: Vec<usize>,
    pub actual_argmax_values: Vec<usize>,
}

pub fn train_model<T>(
    details: DatasetDetails<
        impl UnaryModel,
        impl BinaryOperator,
        impl OptimizerTrait,
        impl TensorPrinter,
    >,
) -> Result<NeuralMachineTestOutput, Error> {
    let train_examples = &details.train_examples;
    let model = details.model;
    let loss_operator = details.loss_operator;
    let maximum_device_streams = 16;
    let device = details.device;
    let clip_grad_norm = details.clip_gradient_norm;
    let shuffle_examples = details.shuffle_examples;
    let batch_size = details.batch_size;
    let optimizer = details.optimizer;
    let mut printer = details.printer;

    let program = NeuralProgram::try_new(
        &device,
        &model,
        &loss_operator,
        &optimizer,
        clip_grad_norm,
        batch_size,
    )?;
    let mut neural_machine = NeuralMachine::<T, DefaultStreamScheduler>::try_new(
        &device,
        program,
        maximum_device_streams,
    )?;

    let train_inputs: Vec<_> = train_examples.iter().map(|x| x.clone().0).collect();
    let train_outputs: Vec<_> = train_examples.iter().map(|x| x.clone().1).collect();

    let epochs = details.epochs;

    print_device_mem_info(&device)?;

    let (initial_metrics, _, _) = print_training_examples(
        0,
        &mut neural_machine,
        &mut printer,
        &train_inputs,
        &train_outputs,
    )?;

    println!("");

    training_loop(
        shuffle_examples,
        batch_size,
        epochs,
        &mut neural_machine,
        &train_inputs,
        &train_outputs,
    )?;

    let (final_metrics, expected_argmax_values, actual_argmax_values) = print_training_examples(
        epochs,
        &mut neural_machine,
        &mut printer,
        &train_inputs,
        &train_outputs,
    )?;

    let output = NeuralMachineTestOutput {
        initial_metrics,
        final_metrics,
        expected_argmax_values,
        actual_argmax_values,
    };

    // Test on test examples.
    let test_examples = details.test_examples;
    for (test_number, (test_input, test_output)) in test_examples.iter().enumerate() {
        let actual_output = neural_machine.infer(&test_input)?;
        let loss = neural_machine.loss(&test_output)?;
        let loss: &Tensor = &loss.tensor();
        let loss: f32 = loss.try_into()?;
        println!("test example: {},  loss: {}", test_number, loss,);

        printer.print_expected_output_and_actual_output(
            &test_input.tensor(),
            &test_output.tensor(),
            &actual_output.tensor(),
        )?;
    }

    Ok(output)
}

fn print_training_examples<T>(
    epoch: usize,
    neural_machine: &mut NeuralMachine<T, DefaultStreamScheduler>,
    printer: &mut impl TensorPrinter,
    inputs: &[TensorWithGrad],
    outputs: &[TensorWithGrad],
) -> Result<(Metrics, Vec<usize>, Vec<usize>), Error> {
    let mut expected_argmax_values = Vec::new();
    let mut actual_argmax_values = Vec::new();
    let last_row = outputs[0].tensor().rows() - 1;

    let mut total_loss = 0.0;
    for i in 0..inputs.len() {
        let input = &inputs[i];
        let expected_output = &outputs[i];
        let actual_output = neural_machine.infer(input)?;

        let loss = neural_machine.loss(expected_output)?;
        let loss: &Tensor = &loss.tensor();
        let loss: f32 = loss.try_into()?;
        total_loss += loss;

        let actual_output = &actual_output.tensor();

        let expected_output = &outputs[i].tensor();
        let expected_output_argmaxes = get_row_argmaxes(expected_output)?;
        let expected_argmax = expected_output_argmaxes[last_row].to_owned();
        expected_argmax_values.push(expected_argmax);

        let actual_output_argmaxes = get_row_argmaxes(actual_output)?;
        let actual_argmax = actual_output_argmaxes[last_row].to_owned();
        actual_argmax_values.push(actual_argmax);

        println!("----");
        println!("  epoch: {}, example: {}, loss: {}", epoch, i, loss,);

        printer.print_expected_output_and_actual_output(
            &input.tensor(),
            expected_output,
            actual_output,
        )?;
    }

    let metrics = Metrics { total_loss };

    Ok((metrics, expected_argmax_values, actual_argmax_values))
}

pub fn get_row_argmaxes(tensor: &Tensor) -> Result<Vec<usize>, Error> {
    let values = tensor.get_values()?;
    let cols = tensor.cols();
    let mut argmaxes = vec![];
    for row in 0..tensor.rows() {
        let mut argmax_col = 0;
        for col in 0..cols {
            let acc = values[tensor.index(row, argmax_col)];
            let item = values[tensor.index(row, col)];
            if item > acc {
                argmax_col = col;
            }
        }
        argmaxes.push(argmax_col);
    }
    Ok(argmaxes)
}

pub fn get_row_argmax(tensor: &Tensor, row: usize) -> Result<usize, Error> {
    let values = tensor.get_values()?;
    let cols = tensor.cols();
    let mut argmax_col = 0;
    for col in 0..cols {
        let acc = values[tensor.index(row, argmax_col)];
        let item = values[tensor.index(row, col)];
        if item > acc {
            argmax_col = col;
        }
    }
    Ok(argmax_col)
}

pub fn training_loop<T>(
    shuffle_examples: bool,
    batch_size: usize,
    epochs: usize,
    neural_machine: &mut NeuralMachine<T, DefaultStreamScheduler>,
    inputs: &Vec<TensorWithGrad>,
    outputs: &Vec<TensorWithGrad>,
) -> Result<(), Error> {
    if inputs.len() % batch_size != 0 {
        panic!(
            "Bad batch_size {} for examples count {}",
            batch_size,
            inputs.len()
        );
    }
    let indices = (0..inputs.len()).collect::<Vec<_>>();
    let mut global_step = 0;
    for epoch in 0..epochs {
        let batches = make_batches(&indices, shuffle_examples, batch_size);

        for (batch_id, batch) in batches.iter().enumerate() {
            let mut batch_loss = 0.0;
            neural_machine.enable_dropout()?;
            for i in batch.iter() {
                let input = &inputs[*i];
                let output = &outputs[*i];
                let _output = neural_machine.infer(input)?;
                let loss = neural_machine.loss(output)?;
                let loss: &Tensor = &loss.tensor();
                let loss: f32 = loss.try_into()?;
                batch_loss += loss;
                neural_machine.compute_gradient()?;
            }
            println!(
                "Epoch: {} / {}   batch: {} / {}   global_step: {}   batch_loss: {}",
                epoch + 1,
                epochs,
                batch_id + 1,
                batches.len(),
                global_step + 1,
                batch_loss
            );
            neural_machine.optimize()?;
            global_step += 1;
        }
    }

    neural_machine.disable_dropout()?;
    Ok(())
}

pub fn time_it<F: Fn() -> T, T>(text: &str, f: F) -> T {
    let start = SystemTime::now();
    let result = f();
    let end = SystemTime::now();
    let duration = end.duration_since(start).unwrap();
    println!("{} took {} μs", text, duration.as_micros());
    result
}
