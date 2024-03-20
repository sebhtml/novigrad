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
    let examples = load_examples(Dataset::Simple);
    let input_size = examples[0].0.dimensions()[1];
    let output_size = examples[0].1.dimensions()[1];

    let layers = vec![
        LayerConfig {
            rows: 16,
            cols: input_size,
            activation: Activation::Sigmoid,
        },
        LayerConfig {
            rows: 16,
            cols: 16,
            activation: Activation::Sigmoid,
        },
        LayerConfig {
            rows: output_size,
            cols: 16,
            activation: Activation::Softmax,
        },
    ];

    let mut network = Network::new(layers);

    let inputs = examples.iter().map(|x| x.clone().0).collect();
    let outputs = examples.iter().map(|x| x.clone().1).collect();

    let mut last_total_error = f32::NAN;
    let epochs = 10000;
    for epoch in 0..epochs {
        if epoch % 1000 == 0 {
            last_total_error =
                print_total_error(&network, &inputs, &outputs, last_total_error, epoch);
        }
        network.train(&inputs, &outputs);
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

    println!("");
    println!("Final weights");

    let mut total_parameters = 0;
    for (index, layer) in network.layers.iter().enumerate() {
        println!("Layer {}", index);
        println!("Weights {}", layer.weights().as_ref().borrow());
        let mut layer_parameters = 1;
        for dimension in layer.weights().as_ref().borrow().dimensions() {
            layer_parameters *= dimension;
        }
        total_parameters += layer_parameters;
    }
    println!("Total_parameters: {}", total_parameters);
}
