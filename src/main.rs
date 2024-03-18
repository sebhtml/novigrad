// TODO move those module declarations elsewhere, maybe in a module.
mod network;
use network::*;
mod tensor;
use tensor::*;
mod activation;
use activation::*;
mod dataset;
use dataset::*;

fn main() {
    let examples = load_simple_examples();
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
    for i in 0..32000 {
        if i % 1000 == 0 {
            let total_error = network.total_error(&inputs, &outputs);
            let change = (total_error - last_total_error) / last_total_error;
            println!(
                "Iteration {} Total_error {}, change: {}",
                i, total_error, change
            );
            last_total_error = total_error;
        }
        network.train(&inputs, &outputs);
    }

    _ = network.predict_many(&inputs);

    println!("");
    println!("Final weights");

    let mut total_parameters = 0;
    for (index, layer) in network.layers.iter().enumerate() {
        println!("Layer {}", index);
        println!("Weights {}", layer.weights);
        let mut layer_parameters = 1;
        for dimension in layer.weights.dimensions() {
            layer_parameters *= dimension;
        }
        total_parameters += layer_parameters;
    }
    println!("Total_parameters: {}", total_parameters);
}
