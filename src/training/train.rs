use std::ops::Deref;

use crate::{
    total_loss, train, DatasetDetails, Device, Error, GradientDescent, OptimizerTrait, Program,
    Tensor, TensorF32, Tokenizer, TokenizerTrait, UnaryOperator,
};

pub fn print_expected_output_and_actual_output(
    epoch: usize,
    tokenizer: &mut Tokenizer,
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
    println!("Loss {}", loss);

    println!("  input_text: {}", tokenizer.decode(&input_tokens)?);

    println!(
        "  expected_output_text: {}",
        tokenizer.decode(&[expected_output_token])?
    );

    println!(
        "  actual_output_text: {}",
        tokenizer.decode(&[actual_output_token])?
    );

    println!("  input_tokens: {:?}", &input_tokens);
    println!(
        "  epoch: {}, example: {}, loss: {}, expected_output_token: {}, actual_output_token: {}",
        epoch, example, loss, expected_output_token, actual_output_token
    );

    println!("expected_output {}", expected_output);
    println!("actual_output {}", actual_output);

    Ok(())
}

fn print_device_mem_info(device: &Device) -> Result<(), Error> {
    let mem_info = &device.get_memory_info()?;
    println!(
        "Device memory  used: {}, free: {}, total: {}, model_parameters: {}",
        mem_info.used, mem_info.free, mem_info.total, mem_info.model_parameters,
    );
    Ok(())
}

fn print_total_loss(
    device: &Device,
    program: &Program,
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

pub fn train_network_on_dataset(
    dataset_details: DatasetDetails,
) -> Result<NetworkTestOutput, Error> {
    let mut initial_total_error = f32::NAN;
    let examples = &dataset_details.examples;
    let learning_rate = dataset_details.learning_rate;
    let program = dataset_details.program;
    let mut tokenizer = dataset_details.tokenizer;
    let device = dataset_details.device;
    let optimizer: Box<dyn OptimizerTrait> = Box::new(GradientDescent::default());
    let inputs: Vec<_> = examples.iter().map(|x| x.clone().0).collect();
    let outputs: Vec<_> = examples.iter().map(|x| x.clone().1).collect();

    let mut last_total_error = f32::NAN;
    let epochs = dataset_details.epochs;
    let progress = dataset_details.progress;

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
    program: &Program,
    tokenizer: &mut Tokenizer,
    inputs: &[Tensor],
    outputs: &[Tensor],
) -> Result<(Vec<usize>, Vec<usize>), Error> {
    let mut expected_argmax_values = Vec::new();
    let mut actual_argmax_values = Vec::new();

    for i in 0..inputs.len() {
        let input = &inputs[i];
        let actual_output = program.forward(&input)?;
        let loss = program.loss(&outputs[i])?;
        let loss: &TensorF32 = &loss.tensor().deref().borrow();
        let loss: f32 = loss.try_into()?;

        let expected_output: &TensorF32 = &outputs[i].tensor().deref().borrow();
        let expected_output_argmaxes = get_row_argmaxes(expected_output)?;
        let expected_argmax = expected_output_argmaxes.last().unwrap().to_owned();
        expected_argmax_values.push(expected_argmax);

        let actual_output: &TensorF32 = &actual_output.tensor().deref().borrow();
        let actual_output_argmaxes = get_row_argmaxes(actual_output)?;
        let actual_argmax = actual_output_argmaxes.last().unwrap().to_owned();
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
