use crate::{DatasetDetails, Error, Network, Tensor, TrainWorkingMemory};

pub fn print_expected_output_and_actual_output(
    example: usize,
    expected_output: &Tensor,
    actual_output: &Tensor,
    expected_argmax: usize,
    actual_argmax: usize,
    loss: Option<&Tensor>,
) {
    let cols = expected_output.cols();
    let last_row = actual_output.rows() - 1;

    println!("----");
    println!("Example {}", example);
    println!(
        "expected_argmax {}, actual_argmax {}",
        expected_argmax, actual_argmax
    );
    println!("");
    for col in 0..cols {
        let loss = match loss {
            Some(loss) => loss.get(0, col),
            _ => Default::default(),
        };
        println!(
            "index {}  expected {}  actual {}  loss {}",
            col,
            expected_output.get(0, col),
            actual_output.get(last_row, col),
            loss
        );
    }
}

fn print_total_error(
    network: &Network,
    inputs: &Vec<Tensor>,
    outputs: &Vec<Tensor>,
    last_total_error: f32,
    epoch: usize,
) -> Result<f32, Error> {
    let total_error = network.total_error(inputs, outputs)?;
    let change = (total_error - last_total_error) / last_total_error;
    println!(
        "Epoch {} Total_error {}, change: {}",
        epoch, total_error, change
    );
    Ok(total_error)
}

pub struct NetworkTestOutput {
    pub initial_total_error: f32,
    pub final_total_error: f32,
    pub expected_argmax_values: Vec<usize>,
    pub actual_argmax_values: Vec<usize>,
}

pub fn train_network_on_dataset(
    dataset_details: &DatasetDetails,
) -> Result<NetworkTestOutput, Error> {
    let mut train_working_memory = TrainWorkingMemory::default();
    let mut initial_total_error = f32::NAN;
    let examples = &dataset_details.examples;
    let layers = &dataset_details.layers;
    let loss_function_name = &dataset_details.loss_function_name;

    let mut network = Network::new(layers, loss_function_name);

    let inputs = examples.iter().map(|x| x.clone().0).collect();
    let outputs = examples.iter().map(|x| x.clone().1).collect();

    let mut last_total_error = f32::NAN;
    let epochs = dataset_details.epochs;
    let progress = dataset_details.progress;
    for epoch in 0..epochs {
        if epoch % progress == 0 {
            let total_error =
                print_total_error(&network, &inputs, &outputs, last_total_error, epoch)?;
            if epoch == 0 {
                initial_total_error = total_error;
            }
            last_total_error = total_error;
        }
        network.train(&mut train_working_memory, epoch, &inputs, &outputs);
    }
    let final_total_error =
        print_total_error(&network, &inputs, &outputs, last_total_error, epochs)?;

    let predictions = network.predict_many(&inputs);

    let mut expected_argmax_values = Vec::new();
    let mut actual_argmax_values = Vec::new();

    for i in 0..inputs.len() {
        let expected_output = &outputs[i];
        let actual_output = &predictions[i];

        let cols = expected_output.cols();
        let mut expected_argmax = 0;
        for col in 0..cols {
            if expected_output.get(0, col) > expected_output.get(0, expected_argmax) {
                expected_argmax = col;
            }
        }

        let last_row = actual_output.rows() - 1;
        let mut actual_argmax = 0;
        for col in 0..cols {
            if actual_output.get(last_row, col) > actual_output.get(last_row, actual_argmax) {
                actual_argmax = col;
            }
        }

        expected_argmax_values.push(expected_argmax);
        actual_argmax_values.push(actual_argmax);

        print_expected_output_and_actual_output(
            i,
            expected_output,
            actual_output,
            expected_argmax,
            actual_argmax,
            None,
        );
    }

    let output = NetworkTestOutput {
        initial_total_error,
        final_total_error,
        expected_argmax_values,
        actual_argmax_values,
    };
    Ok(output)
}
