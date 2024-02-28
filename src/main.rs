mod network;
mod neuron;
use network::*;
use neuron::*;

fn main() {
    let inputs = vec![vec![42.0], vec![41.0]];
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
}
