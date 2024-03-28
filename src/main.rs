// TODO move those module declarations elsewhere, maybe in a module.
mod network;
use network::{train::train_network_on_dataset, *};
mod tensor;
use tensor::*;
mod activation;
use activation::*;
mod layer;
use layer::*;
mod dataset;
use dataset::*;
mod loss;

fn main() {
    let dataset = Dataset::Simple;
    //let dataset = Dataset::MegaMan;
    train_network_on_dataset(&dataset).expect("Ok");
}
