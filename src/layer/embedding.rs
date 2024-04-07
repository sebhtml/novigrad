use crate::{DeltaWorkingMemory, Error, Layer, LayerType, Tensor, TensorTrait};
use core::mem::swap;
use rand::{distributions::Uniform, thread_rng, Rng};

pub struct Embedding {
    embedding_table: Tensor,
    embedding_table_delta: Tensor,
    has_pending_change: bool,
    activation_tensor: Tensor,
}

impl Embedding {
    pub fn new(_hidden_dimensions: usize) -> Self {
        Self {
            embedding_table: get_u8_embedding_table(),
            embedding_table_delta: Default::default(),
            has_pending_change: Default::default(),
            activation_tensor: Default::default(),
        }
    }
}

impl Layer for Embedding {
    fn plan_change(
        &mut self,
        learning_rate: f32,
        previous_activation: &Tensor,
        layer_delta: &Tensor,
    ) {
        // TODO
    }

    fn commit_change(&mut self) -> Result<(), Error> {
        // TODO
        return Ok(());
        if !self.has_pending_change {
            return Ok(());
        }

        let mut addition = Tensor::default();
        {
            let embedding_table_delta = &self.embedding_table_delta;
            let embedding_table = &self.embedding_table;
            let op_result = embedding_table.sub(embedding_table_delta, &mut addition);
            op_result.expect("Ok");
        }

        let embedding_table = &mut self.embedding_table;
        swap(embedding_table, &mut addition);

        self.has_pending_change = false;
        Ok(())
    }

    fn forward(&mut self, input: &Tensor) -> Result<(), Error> {
        Tensor::matmul(
            input,
            &self.embedding_table,
            &mut self.activation_tensor,
            Default::default(),
        )
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
        next_layer: Option<&LayerType>,
        next_layer_delta: &Tensor,
        _using_softmax_and_cross_entropy_loss: bool,
        layer_delta: &mut Tensor,
    ) {
        match next_layer {
            Some(next_layer) => {
                // Hidden layer
                next_layer.backward(next_layer_delta, layer_delta);
            }
            None => panic!("Not implemented"),
        }
    }
}

pub struct EmbeddingConfig {
    pub hidden_dimensions: usize,
}

impl Into<Embedding> for &EmbeddingConfig {
    fn into(self) -> Embedding {
        Embedding::new(self.hidden_dimensions)
    }
}

fn get_u8_embedding_table() -> Tensor {
    let mut rng = thread_rng();
    let mut embeddings_table: Vec<f32> = Vec::new();
    let left = 0.0;
    let right = 1.0;
    let number_of_different_tokens = 256;
    let width = 256;
    let uniform = Uniform::new(left, right);

    let mut token = 0;
    while token < number_of_different_tokens {
        let mut token_embeddings: Vec<f32> = Vec::new();
        for _ in 0..width {
            let value = rng.sample(uniform);
            token_embeddings.push(value);
        }
        embeddings_table.append(&mut token_embeddings);
        token += 1;
    }
    Tensor::new(width, width, embeddings_table)
}
