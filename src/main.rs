mod network;
use std::os::linux::net;

use network::*;
mod matrix;
use matrix::*;
mod activation;
use activation::*;

fn main() {
    let examples = vec![
        (vec![1.0, 2.0, 3.0, 4.0], vec![0.0]),
        (vec![1.0, 2.0, 2.0, 4.0], vec![0.0]),
        (vec![2.0, 2.0, 3.0, 4.0], vec![0.0]),
        (vec![1.0, 2.0, 4.0, 4.0], vec![0.0]),
        (vec![11.0, 12.0, 13.0, 14.0], vec![1.0]),
        (vec![11.0, 12.0, 13.0, 14.0], vec![1.0]),
        (vec![11.0, 12.0, 14.0, 14.0], vec![1.0]),
        (vec![10.0, 12.0, 13.0, 14.0], vec![1.0]),
        (vec![11.0, 12.0, 13.0, 15.0], vec![1.0]),
    ];
    let inputs = examples.iter().map(|x| x.clone().0).collect();
    let outputs = examples.iter().map(|x| x.clone().1).collect();

    let mut network = Network::new();

    let mut last_total_error = f32::NAN;
    for i in 0..10000 {
        if i % 100 == 0 {
            let total_error = network.total_error(&inputs, &outputs);
            let change = (total_error - last_total_error) / last_total_error;
            println!("Total_error {}, change: {}", total_error, change);
            last_total_error = total_error;
        }
        println!("Training iteration {}", i);
        network.train(&inputs, &outputs);
    }

    _ = network.predict_many(&inputs);
}
