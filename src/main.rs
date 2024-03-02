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
    let _ = network.predict_many(&inputs);

    println!("Some training !");
    for _ in vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10].iter() {
        network.train(&inputs, &outputs);
    }

    let _ = network.predict_many(&inputs);
}
