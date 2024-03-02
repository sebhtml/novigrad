mod network;
use network::*;
mod matrix;
use matrix::Matrix;
mod activation;
use activation::sigmoid;

fn main() {
    let inputs = vec![vec![42.0], vec![20.0]];
    let outputs = vec![vec![1.0], vec![0.0]];

    let network = Network::default();
    let predicted_outputs = network.predict_many(&inputs);

    for (index, _) in predicted_outputs.iter().enumerate() {
        let input = &inputs[index];
        let output = &outputs[index];
        let predicted_output = &predicted_outputs[index];
        println!(
            "Input: {:?}  Output: {:?}  PredictedOutput: {:?}",
            input, output, predicted_output
        );
    }

    println!("Some training !");
    for _ in vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10].iter() {
        network.train(&inputs, &outputs);
    }

    for (index, _) in predicted_outputs.iter().enumerate() {
        let input = &inputs[index];
        let output = &outputs[index];
        let predicted_output = &predicted_outputs[index];
        println!(
            "Input: {:?}  Output: {:?}  PredictedOutput: {:?}",
            input, output, predicted_output
        );
    }
}
