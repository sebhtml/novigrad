mod network;
use network::*;
mod matrix;
use matrix::Matrix;
mod activation;
use activation::sigmoid;

fn main() {
    let inputs = vec![vec![42.0], vec![20.0]];
    let outputs = vec![vec![1.0], vec![0.0]];

    let network = Network::new();

    for i in 0..100 {
        println!("Training iteration {}", i);
        network.train(&inputs, &outputs);
    }

    _ = network.predict_many(&inputs);
}
