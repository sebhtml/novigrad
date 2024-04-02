use crate::{DeltaWorkingMemory, Error, Layer, Tensor};

pub struct Embedding {
    activation_tensor: Tensor,
}

impl Embedding {
    pub fn new(_hidden_dimensions: usize) -> Self {
        // TODO
        Self {
            activation_tensor: Default::default(),
        }
    }
}

impl Layer for Embedding {
    fn plan_change(
        &mut self,
        _learning_rate: f32,
        _previous_activation: &Tensor,
        _layer_delta: &Tensor,
    ) {
        // TODO
    }

    fn commit_change(&mut self) -> Result<(), Error> {
        // TODO
        Ok(())
    }

    fn forward(&mut self, input: &Tensor) -> Result<(), Error> {
        // TODO
        self.activation_tensor.assign(input);
        Ok(())
    }

    fn get_activation_tensor<'a>(&'a self) -> &'a Tensor {
        &self.activation_tensor
    }

    fn backward(&self, _layer_delta: &Tensor, _output_diff: &mut Tensor) {
        panic!("Embedding can not go backward !");
    }

    fn get_layer_delta(
        &self,
        _working_memory: &mut DeltaWorkingMemory,
        _next_layer: Option<&Box<dyn Layer>>,
        _next_layer_delta: &Tensor,
        _using_softmax_and_cross_entropy_loss: bool,
        layer_delta: &mut Tensor,
    ) {
        // TODO
        let new_rows = self.activation_tensor.rows();
        let new_cols = self.activation_tensor.cols();
        layer_delta.reshape(new_rows, new_cols);
    }
}

pub struct EmbeddingConfig {
    pub hidden_dimensions: usize,
}

impl Into<Box<dyn Layer>> for &EmbeddingConfig {
    fn into(self) -> Box<dyn Layer> {
        Box::new(Embedding::new(self.hidden_dimensions))
    }
}
