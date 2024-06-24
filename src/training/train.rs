use rand::seq::SliceRandom;
use rand::thread_rng;
use std::time::SystemTime;

use crate::{
    datasets::DatasetDetails,
    display::TensorPrinter,
    neural_program::NeuralProgram,
    perplexity::get_perplexity,
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
    pub total_perplexity: f32,
}

fn print_metrics(epoch: usize, metrics: &Metrics, previous_metrics: &Metrics) -> Result<(), Error> {
    let total_loss = metrics.total_loss;
    let previous_total_loss = previous_metrics.total_loss;
    let total_loss_change = (total_loss - previous_total_loss) / previous_total_loss;
    println!("----",);
    println!(
        "Epoch {} total_loss {}, change: {}",
        epoch, total_loss, total_loss_change
    );
    let total_perplexity = metrics.total_perplexity;
    let previous_total_perplexity = previous_metrics.total_perplexity;
    let total_perplexity_change =
        (total_perplexity - previous_total_perplexity) / previous_total_perplexity;
    println!(
        "Epoch {} total_perplexity {}, change: {}",
        epoch, total_perplexity, total_perplexity_change
    );
    Ok(())
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
    let mut initial_metrics = Metrics {
        total_loss: f32::NAN,
        total_perplexity: f32::NAN,
    };
    let mut previous_metrics = Metrics {
        total_loss: f32::NAN,
        total_perplexity: f32::NAN,
    };
    let train_examples = &details.train_examples;
    let model = details.model;
    let loss_operator = details.loss_operator;
    let maximum_device_streams = 16;
    let device = details.device;
    let clipped_gradient_norm = details.clipped_gradient_norm;
    let shuffle_examples = details.shuffle_examples;
    let optimizer = details.optimizer;
    let mut printer = details.printer;

    let program = NeuralProgram::try_new(
        &device,
        &model,
        &loss_operator,
        &optimizer,
        clipped_gradient_norm,
    )?;
    let mut neural_machine = NeuralMachine::<T, DefaultStreamScheduler>::try_new(
        &device,
        program,
        maximum_device_streams,
    )?;

    let train_inputs: Vec<_> = train_examples.iter().map(|x| x.clone().0).collect();
    let train_outputs: Vec<_> = train_examples.iter().map(|x| x.clone().1).collect();

    let epochs = details.epochs;
    let progress = details.progress;

    let (_, _) = print_results(
        0,
        &mut neural_machine,
        &mut printer,
        &train_inputs,
        &train_outputs,
    )?;

    for epoch in 0..epochs {
        if epoch % progress == 0 {
            let metrics = total_metrics(&mut neural_machine, &train_inputs, &train_outputs)?;
            print_metrics(epoch, &metrics, &previous_metrics)?;
            print_device_mem_info(&device)?;
            if epoch == 0 {
                initial_metrics = metrics.clone();
            }
            previous_metrics = metrics.clone();
        }
        train(
            &mut neural_machine,
            shuffle_examples,
            &train_inputs,
            &train_outputs,
        )?;
    }
    let final_metrics = total_metrics(&mut neural_machine, &train_inputs, &train_outputs)?;
    print_metrics(epochs, &final_metrics, &previous_metrics)?;
    print_device_mem_info(&device)?;

    let (expected_argmax_values, actual_argmax_values) = print_results(
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
        println!("test example {test_number}  loss {loss}");
        printer.print_expected_output_and_actual_output(
            &test_input.tensor(),
            &test_output.tensor(),
            &actual_output.tensor(),
        )?;
    }

    Ok(output)
}

fn print_results<T>(
    epoch: usize,
    neural_machine: &mut NeuralMachine<T, DefaultStreamScheduler>,
    printer: &mut impl TensorPrinter,
    inputs: &[TensorWithGrad],
    outputs: &[TensorWithGrad],
) -> Result<(Vec<usize>, Vec<usize>), Error> {
    let mut expected_argmax_values = Vec::new();
    let mut actual_argmax_values = Vec::new();
    let last_row = outputs[0].tensor().rows() - 1;

    for i in 0..inputs.len() {
        let input = &inputs[i];
        let expected_output = &outputs[i];
        let actual_output = neural_machine.infer(input)?;

        let loss = neural_machine.loss(expected_output)?;
        let loss: &Tensor = &loss.tensor();
        let loss: f32 = loss.try_into()?;

        let actual_output = &actual_output.tensor();
        let perplexity = get_perplexity(actual_output, actual_output.rows() - 1)?;

        let expected_output = &outputs[i].tensor();
        let expected_output_argmaxes = get_row_argmaxes(expected_output)?;
        let expected_argmax = expected_output_argmaxes[last_row].to_owned();
        expected_argmax_values.push(expected_argmax);

        let actual_output_argmaxes = get_row_argmaxes(actual_output)?;
        let actual_argmax = actual_output_argmaxes[last_row].to_owned();
        actual_argmax_values.push(actual_argmax);

        println!("----");
        println!(
            "  epoch: {}, example: {}, loss: {}, next_token_perplexity: {}, ",
            epoch, i, loss, perplexity,
        );

        printer.print_expected_output_and_actual_output(
            &input.tensor(),
            expected_output,
            actual_output,
        )?;
    }

    Ok((expected_argmax_values, actual_argmax_values))
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

pub fn train<T>(
    neural_machine: &mut NeuralMachine<T, DefaultStreamScheduler>,
    shuffle_examples: bool,
    inputs: &Vec<TensorWithGrad>,
    outputs: &Vec<TensorWithGrad>,
) -> Result<(), Error> {
    let mut indices: Vec<usize> = (0..inputs.len()).collect();
    if shuffle_examples {
        indices.shuffle(&mut thread_rng());
    }
    for i in indices.into_iter() {
        train_with_one_example(neural_machine, &inputs[i], &outputs[i])?;
    }
    Ok(())
}

pub fn total_metrics<T>(
    neural_machine: &mut NeuralMachine<T, DefaultStreamScheduler>,
    inputs: &[TensorWithGrad],
    outputs: &[TensorWithGrad],
) -> Result<Metrics, Error> {
    let mut total_loss = 0.0;
    let mut total_perplexity = 0.0;
    for i in 0..inputs.len() {
        let expected_output = &outputs[i];
        let actual_output = neural_machine.infer(&inputs[i])?;

        // Loss
        let example_loss = neural_machine.loss(expected_output)?;
        let example_loss: &Tensor = &example_loss.tensor();
        let example_loss: f32 = example_loss.try_into()?;
        total_loss += example_loss;

        // Perplexity
        let actual_output = &actual_output.tensor();
        let perplexity = get_perplexity(actual_output, actual_output.rows() - 1)?;
        total_perplexity += perplexity;
    }

    let metrics = Metrics {
        total_loss,
        total_perplexity,
    };
    Ok(metrics)
}

fn train_with_one_example<T>(
    neural_machine: &mut NeuralMachine<T, DefaultStreamScheduler>,
    input: &TensorWithGrad,
    output: &TensorWithGrad,
) -> Result<(), Error> {
    let _output = neural_machine.infer(input)?;
    let _loss = neural_machine.loss(output)?;
    neural_machine.compute_gradient()?;
    neural_machine.optimize()?;

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
