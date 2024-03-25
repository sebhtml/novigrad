// TODO move those module declarations elsewhere, maybe in a module.
mod network;
use network::*;
mod tensor;
use tensor::*;
mod activation;
use activation::*;
mod layer;
use layer::*;
mod dataset;
use crate::tests::test_network_on_dataset;
use dataset::*;

fn main() {
    let dataset = Dataset::Simple;
    //let dataset = Dataset::MegaMan;
    test_network_on_dataset(&dataset);
}
