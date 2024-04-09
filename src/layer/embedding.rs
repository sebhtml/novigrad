use crate::{
    DeltaWorkingMemory, DifferentiableTensor, Error, Layer, Tensor, TRANSPOSE_LHS, TRANSPOSE_RESULT,
};
use rand::{distributions::Uniform, thread_rng, Rng};

pub struct Embedding {
    embedding_table: DifferentiableTensor,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        Self {
            embedding_table: get_embedding_table(num_embeddings, embedding_dim).into(),
        }
    }
}

impl Layer for Embedding {
    fn compute_gradient(&mut self, layer_input: &Tensor, layer_output_delta: &Tensor) {
        let op_result = Tensor::matmul(
            layer_output_delta,
            layer_input,
            &mut self.embedding_table.gradient,
            TRANSPOSE_LHS | TRANSPOSE_RESULT,
        );
        op_result.expect("Ok");
        self.embedding_table.has_gradient = true;
    }

    fn commit_change(&mut self, learning_rate: f32) -> Result<(), Error> {
        self.embedding_table.commit_change(learning_rate);
        Ok(())
    }

    fn forward(&mut self, input: &Tensor, output: &mut Tensor) -> Result<(), Error> {
        debug_assert_eq!(input.cols(), self.embedding_table.tensor.rows());
        Tensor::matmul(
            input,
            &self.embedding_table.tensor,
            output,
            Default::default(),
        )
    }

    fn backward(&self, _layer_delta: &Tensor, _previous_layer_delta: &mut Tensor) {
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
