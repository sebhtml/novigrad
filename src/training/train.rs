use rand::seq::SliceRandom;
use rand::thread_rng;
use std::{ops::Deref, time::SystemTime};

use crate::{
    Device, Error, GradientDescent, ModelDetails, NeuralMachine, OptimizerTrait, Tensor, TensorF32,
    Tokenizer, TokenizerTrait,
};

trait IsPrintable {
    fn is_printable(&self) -> bool;
}

impl IsPrintable for char {
    fn is_printable(&self) -> bool {
        let code = *self as usize;
        if (code >= 32 && code <= 126) || code == 9 || code == 10 || code == 13 {
            return true;
        }
        return false;
    }
}

fn as_printable(output: String, replacement: char) -> String {
    let mut printable: String = String::new();
    for char in output.as_str().chars() {
        if char.is_printable() {
            printable += String::from(char).as_str();
        } else {
            printable += String::from(replacement).as_str();
        }
    }
    printable
}

fn tokens_to_text(
    input_tokens: &[usize],
    tokenizer: &mut Option<Tokenizer>,
) -> Result<String, Error> {
    let input_text = match tokenizer {
        Some(tokenizer) => tokenizer.decode(&input_tokens)?,
        None => input_tokens
            .to_vec()
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", "),
    };
    Ok(input_text)
}

pub fn print_expected_output_and_actual_output(
    epoch: usize,
    tokenizer: &mut Option<Tokenizer>,
    example: usize,
    input: &TensorF32,
    expected_output: &TensorF32,
    actual_output: &TensorF32,
    expected_output_token: usize,
    actual_output_token: usize,
    loss: f32,
) -> Result<(), Error> {
    let input_tokens = get_row_argmaxes(input)?;

    println!("----");
    println!("Epoch {} Example {}", epoch, example);

    println!(
        "  input_text: {}",
        tokens_to_text(&input_tokens, tokenizer)?
    );

    println!(
        "  expected_output_text: {}",
        tokens_to_text(&[expected_output_token], tokenizer)?
    );

    let actual_output_text: String = tokens_to_text(&[actual_output_token], tokenizer)?;

    println!(
        "  actual_output_text: {}",
        as_printable(actual_output_text, '?'),
    );

    println!("  input_tokens: {:?}", &input_tokens);
    println!(
        "  epoch: {}, example: {}, loss: {}, expected_output_token: {}, actual_output_token: {}",
        epoch, example, loss, expected_output_token, actual_output_token
    );

    if expected_output.cols() < 10 {
        println!("expected_output {}", expected_output);
        println!("actual_output {}", actual_output);
    }

    Ok(())
}

fn print_device_mem_info(device: &Device) -> Result<(), Error> {
    let mem_info = &device.get_memory_info()?;
    println!(
        "Device memory  used: {}, free: {}, total: {}",
        mem_info.used, mem_info.free, mem_info.total,
    );
    Ok(())
}

fn print_total_loss(
    device: &Device,
    program: &NeuralMachine,
    inputs: &Vec<Tensor>,
    outputs: &Vec<Tensor>,
    last_total_loss: f32,
    epoch: usize,
) -> Result<f32, Error> {
    let total_loss = total_loss(program, inputs, outputs)?;
    let change = (total_loss - last_total_loss) / last_total_loss;
    println!("----",);
    println!(
        "Epoch {} Total_loss {}, change: {}",
        epoch, total_loss, change
    );
    print_device_mem_info(device)?;
    Ok(total_loss)
}

pub struct NetworkTestOutput {
    pub initial_total_error: f32,
    pub final_total_error: f32,
    pub expected_argmax_values: Vec<usize>,
    pub actual_argmax_values: Vec<usize>,
}

pub fn train_model(details: ModelDetails) -> Result<NetworkTestOutput, Error> {
    let mut initial_total_error = f32::NAN;
    let examples = &details.examples;
    let learning_rate = details.learning_rate;
    let model = details.model;
    let loss_operator = details.loss_operator;
    let clipped_gradient_norm = details.clipped_gradient_norm;
    let mut tokenizer = details.tokenizer;
    let device = details.device;
    let optimizer: Box<dyn OptimizerTrait> = Box::new(GradientDescent::default());
    let program = NeuralMachine::try_new(&device, &model, &loss_operator, clipped_gradient_norm)?;

    let inputs: Vec<_> = examples.iter().map(|x| x.clone().0).collect();
    let outputs: Vec<_> = examples.iter().map(|x| x.clone().1).collect();

    let mut last_total_error = f32::NAN;
    let epochs = details.epochs;
    let progress = details.progress;

    let (_, _) = print_results(0, &program, &mut tokenizer, &inputs, &outputs)?;

    for epoch in 0..epochs {
        if epoch % progress == 0 {
            let total_error = print_total_loss(
                &device,
                &program,
                &inputs,
                &outputs,
                last_total_error,
                epoch,
            )?;
            if epoch == 0 {
                initial_total_error = total_error;
            }
            last_total_error = total_error;
            if last_total_error == 0.0 {
                break;
            }
        }
        train(
            &program,
            &device,
            &optimizer,
            learning_rate,
            epoch,
            &inputs,
            &outputs,
        )?;
    }
    let final_total_error = print_total_loss(
        &device,
        &program,
        &inputs,
        &outputs,
        last_total_error,
        epochs,
    )?;

    let (expected_argmax_values, actual_argmax_values) =
        print_results(epochs, &program, &mut tokenizer, &inputs, &outputs)?;

    let output = NetworkTestOutput {
        initial_total_error,
        final_total_error,
        expected_argmax_values,
        actual_argmax_values,
    };
    Ok(output)
}

fn print_results(
    epoch: usize,
    program: &NeuralMachine,
    tokenizer: &mut Option<Tokenizer>,
    inputs: &[Tensor],
    outputs: &[Tensor],
) -> Result<(Vec<usize>, Vec<usize>), Error> {
    let mut expected_argmax_values = Vec::new();
    let mut actual_argmax_values = Vec::new();
    let last_row = outputs[0].tensor().deref().borrow().rows() - 1;

    for i in 0..inputs.len() {
        let input = &inputs[i];
        let expected_output = &outputs[i];
        let actual_output = program.forward(input, expected_output)?;
        let loss = program.loss()?;
        let loss: &TensorF32 = &loss.tensor().deref().borrow();
        let loss: f32 = loss.try_into()?;

        let expected_output: &TensorF32 = &outputs[i].tensor().deref().borrow();
        let expected_output_argmaxes = get_row_argmaxes(expected_output)?;
        let expected_argmax = expected_output_argmaxes[last_row].to_owned();
        expected_argmax_values.push(expected_argmax);

        let actual_output: &TensorF32 = &actual_output.tensor().deref().borrow();
        let actual_output_argmaxes = get_row_argmaxes(actual_output)?;
        let actual_argmax = actual_output_argmaxes[last_row].to_owned();
        actual_argmax_values.push(actual_argmax);

        print_expected_output_and_actual_output(
            epoch,
            tokenizer,
            i,
            &input.tensor().deref().borrow(),
            expected_output,
            actual_output,
            expected_argmax,
            actual_argmax,
            loss,
        )?;
    }

    Ok((expected_argmax_values, actual_argmax_values))
}

fn get_row_argmaxes(tensor: &TensorF32) -> Result<Vec<usize>, Error> {
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

pub fn train(
    program: &NeuralMachine,
    device: &Device,
    optimizer: &Box<dyn OptimizerTrait>,
    learning_rate: f32,
    epoch: usize,
    inputs: &Vec<Tensor>,
    outputs: &Vec<Tensor>,
) -> Result<(), Error> {
    let mut indices: Vec<usize> = (0..inputs.len()).collect();
    indices.shuffle(&mut thread_rng());
    for i in indices.into_iter() {
        train_with_one_example(
            program,
            device,
            optimizer,
            learning_rate,
            epoch,
            i,
            &inputs[i],
            &outputs[i],
        )?;
    }
    Ok(())
}

pub fn total_loss(
    program: &NeuralMachine,
    inputs: &[Tensor],
    outputs: &[Tensor],
) -> Result<f32, Error> {
    let mut total_error = 0.0;
    for i in 0..inputs.len() {
        let expected_output = &outputs[i];
        let _ = program.forward(&inputs[i], expected_output)?;
        let example_loss = program.loss()?;
        let example_loss: &TensorF32 = &example_loss.tensor().deref().borrow();
        let example_loss: f32 = example_loss.try_into()?;
        total_error += example_loss;
    }

    Ok(total_error)
}

fn train_with_one_example(
    program: &NeuralMachine,
    device: &Device,
    optimizer: &Box<dyn OptimizerTrait>,
    learning_rate: f32,
    _epoch: usize,
    _example_index: usize,
    x: &Tensor,
    y: &Tensor,
) -> Result<(), Error> {
    let _output = program.forward(x, y)?;
    let _loss = program.loss()?;
    let gradients = device.tensors_to_optimize();
    let gradients: &[Tensor] = &gradients.deref().borrow();

    optimizer.optimize(&gradients, learning_rate)?;

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
