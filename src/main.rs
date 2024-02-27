mod network;
mod neuron;
use network::*;
use neuron::*;

fn main() {
    let inputs = vec![vec![1.0, 2.0], vec![1.0, 3.0], vec![11.0, 222.0]];

    let network = Network::default();
    let predictions = network.predict_many(&inputs);

    for (index, prediction) in predictions.iter().enumerate() {
        println!("Input: {:#?}  prediction: {:#?}", inputs[index], prediction);
    }
}
