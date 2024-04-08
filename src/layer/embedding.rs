use crate::{
    DeltaWorkingMemory, Error, Layer, LayerType, Tensor, TensorTrait, TRANSPOSE_LHS,
    TRANSPOSE_RESULT,
};
use core::mem::swap;
use rand::{distributions::Uniform, thread_rng, Rng};

pub struct Embedding {
    embedding_table: Tensor,
    embedding_table_delta: Tensor,
    has_pending_change: bool,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        Self {
            embedding_table: get_embedding_table(num_embeddings, embedding_dim),
            embedding_table_delta: Default::default(),
            has_pending_change: Default::default(),
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
        let mut tmp = Tensor::default();
        let op_result = Tensor::matmul(
            layer_delta,
            previous_activation,
            &mut tmp,
            TRANSPOSE_LHS | TRANSPOSE_RESULT,
        );
        op_result.expect("Ok");
        let op_result = tmp.scalar_mul(-learning_rate, &mut self.embedding_table_delta);
        op_result.expect("Ok");
        self.has_pending_change = true;
    }

    fn commit_change(&mut self) -> Result<(), Error> {
        if !self.has_pending_change {
            return Ok(());
        }

        let mut addition = Tensor::default();
        {
            let embedding_table_delta = &self.embedding_table_delta;
            let embedding_table = &self.embedding_table;
            let op_result = embedding_table.add(embedding_table_delta, &mut addition);
            op_result.expect("Ok");
        }

        let embedding_table = &mut self.embedding_table;
        swap(embedding_table, &mut addition);

        self.has_pending_change = false;
        Ok(())
    }

    fn forward(&mut self, input: &Tensor, output: &mut Tensor) -> Result<(), Error> {
        debug_assert_eq!(input.cols(), self.embedding_table.rows());
        Tensor::matmul(input, &self.embedding_table, output, Default::default())
    }

    fn backward(&self, _layer_delta: &Tensor, _output_diff: &mut Tensor) {
        panic!("Embedding can not go backward !");
    }

    fn get_layer_delta(
        &self,
        _working_memory: &mut DeltaWorkingMemory,
        _layer_input: &Tensor,
        _layer_output: &Tensor,
        back_propagated_delta: &Tensor,
        _is_last_layer: bool,
        layer_delta: &mut Tensor,
    ) {
        layer_delta.assign(back_propagated_delta)
    }
}

pub struct EmbeddingConfig {
    pub num_embeddings: usize,
    pub embedding_dim: usize,
}

impl Into<Embedding> for &EmbeddingConfig {
    fn into(self) -> Embedding {
        Embedding::new(self.num_embeddings, self.embedding_dim)
    }
}

fn get_embedding_table(num_embeddings: usize, embedding_dim: usize) -> Tensor {
    let mut rng = thread_rng();
    let mut embeddings_table: Vec<f32> = Vec::new();
    let left = 0.0;
    let right = 1.0;
    let uniform = Uniform::new(left, right);

    let mut token = 0;
    while token < num_embeddings {
        let mut token_embeddings: Vec<f32> = Vec::new();
        for _ in 0..embedding_dim {
            let value = rng.sample(uniform);
            token_embeddings.push(value);
        }
        embeddings_table.append(&mut token_embeddings);
        token += 1;
    }
    Tensor::new(num_embeddings, embedding_dim, embeddings_table)
}
