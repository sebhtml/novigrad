mod network;

use network::*;
mod matrix;
use matrix::*;
mod activation;
use activation::*;

fn main() {
    let examples = vec![
        (
            //
            vec![1.0, 0.0, 0.0, 0.0], //
            vec![0.0, 0.5],
        ),
        (
            //
            vec![1.0, 0.0, 0.0, 1.0], //
            vec![0.0, 0.5],
        ),
        (
            //
            vec![0.0, 0.0, 1.0, 0.0], //
            vec![1.0, 0.9],
        ),
        (
            //
            vec![0.0, 1.0, 1.0, 0.0], //
            vec![1.0, 0.9],
        ),
    ];

    let inputs = examples.iter().map(|x| x.clone().0).collect();
    let outputs = examples.iter().map(|x| x.clone().1).collect();

    let mut network = Network::new();

    let mut last_total_error = f32::NAN;
    for i in 0..10000 {
        if i % 100 == 0 {
            let total_error = network.total_error(&inputs, &outputs);
            let change = (total_error - last_total_error) / last_total_error;
            println!(
                "Iteration {} Total_error {}, change: {}",
                i, total_error, change
            );
            last_total_error = total_error;
        }
        println!("Training iteration {}", i);
        network.train(&inputs, &outputs);
    }

    _ = network.predict_many(&inputs);
}
