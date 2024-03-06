mod network;
use network::*;
mod matrix;
use matrix::*;
mod activation;
use activation::*;

fn main() {
    let inputs = vec![vec![1.0, 2.0, 3.0, 4.0], vec![11.0, 12.0, 13.0, 14.0]];
    let outputs = vec![vec![0.0], vec![10.0]];

    let mut network = Network::new();

    for i in 0..100 {
        println!("Training iteration {}", i);
        network.train(&inputs, &outputs);
    }

    _ = network.predict_many(&inputs);
}
