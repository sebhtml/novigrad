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
mod gradient;
mod loss;
use gradient::*;
mod blas;

extern crate blas_src;

fn main() {
    //let dataset = Dataset::Simple;
    let dataset = Dataset::MegaMan;
    let dataset_details = load_dataset(&dataset);
    train_network_on_dataset(&dataset_details).expect("Ok");
}
