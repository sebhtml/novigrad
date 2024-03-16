// TODO move those module declarations elsewhere, maybe in a module.
mod network;
use network::*;
mod matrix;
use matrix::*;
mod activation;
use activation::*;
mod sigmoid;
use sigmoid::*;
mod softmax;
use softmax::*;
mod dataset;
use dataset::*;

fn main() {
    let examples = load_simple_examples();

    let layers = vec![
        LayerConfig {
            rows: 4,
            cols: 16,
            activation: Activation::Sigmoid,
        },
        LayerConfig {
            rows: 16,
            cols: 16,
            activation: Activation::Sigmoid,
        },
        LayerConfig {
            rows: 16,
            cols: 2,
            activation: Activation::Softmax,
        },
    ];

    let mut network = Network::new(layers);

    let inputs = examples.iter().map(|x| x.clone().0).collect();
    let outputs = examples.iter().map(|x| x.clone().1).collect();

    let mut last_total_error = f32::NAN;
    for i in 0..100000 {
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
}
