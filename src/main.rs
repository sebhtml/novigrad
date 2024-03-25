// TODO move those module declarations elsewhere, maybe in a module.
mod network;
use network::*;
mod tensor;
use tensor::*;
mod activation;
use activation::*;
mod layer;
use layer::*;
mod dataset;
use dataset::*;

fn print_total_error(
    network: &Network,
    inputs: &Vec<Tensor>,
    outputs: &Vec<Tensor>,
    last_total_error: f32,
    epoch: usize,
) -> f32 {
    let total_error = network.total_error(inputs, outputs);
    let change = (total_error - last_total_error) / last_total_error;
    println!(
        "Epoch {} Total_error {}, change: {}",
        epoch, total_error, change
    );
    total_error
}

fn main() {
    let dataset = Dataset::Simple;
    //let dataset = Dataset::MegaMan;
    let dataset_details = load_dataset(&dataset);
    let examples = dataset_details.examples;
    let layers = dataset_details.layers;

    let mut network = Network::new(layers);

    let inputs = examples.iter().map(|x| x.clone().0).collect();
    let outputs = examples.iter().map(|x| x.clone().1).collect();

    let mut last_total_error = f32::NAN;
    let epochs = 1000000;
    for epoch in 0..epochs {
        if epoch % 10000 == 0 {
            last_total_error =
                print_total_error(&network, &inputs, &outputs, last_total_error, epoch);
        }
        network.train(epoch, &inputs, &outputs);
    }
    _ = print_total_error(&network, &inputs, &outputs, last_total_error, epochs);

    let predictions = network.predict_many(&inputs);

    for i in 0..inputs.len() {
        let output = &outputs[i];
        let prediction = &predictions[i];
        println!("Example {}", i);
        println!("Expected {}", output);
        println!("Actual   {}", prediction);
    }

    /*
    println!("");
    println!("Final weights");

    let mut total_parameters = 0;
    for (index, layer) in network.layers.iter().enumerate() {
        println!("Layer {}", index);
        println!("Weights {}", layer.weights().as_ref().borrow());
        let rows = layer.weights().as_ref().borrow().rows();
        let cols = layer.weights().as_ref().borrow().cols();
        let layer_parameters = rows * cols;
        total_parameters += layer_parameters;
    }
    println!("Total_parameters: {}", total_parameters);
     */
}
