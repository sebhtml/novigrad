use std::rc::Rc;

use crate::{devices::Device, Error, Identity, MatMul, Operator, Tensor, TensorF32};
use rand::{distributions::Uniform, thread_rng, Rng};

/// Embedding is not a ONNX operator.
/// https://onnx.ai/onnx/operators/index.html ???
#[derive(Clone)]
pub struct Embedding {
    embedding_table: Tensor,
    matmul: MatMul,
}

impl Embedding {
    pub fn new(device: &Device, num_embeddings: usize, embedding_dim: usize) -> Self {
        let embedding_table = get_embedding_table(device, num_embeddings, embedding_dim);
        let len = embedding_table.len();
        let mut transposed = device.tensor_f32(embedding_dim, num_embeddings, vec![0.0; len]);
        // TODO don't unwrap directly
        embedding_table.transpose(&mut transposed).unwrap();
        // TODO don't unwrap directly
        let embedding_table = device.tensor(
            Rc::new(Identity::new(device)),
            &vec![],
            transposed.rows(),
            transposed.cols(),
            transposed.get_values().unwrap(),
            true,
            true,
        );

        let transb = true;
        let matmul = MatMul::new(device, transb);

        Self {
            embedding_table,
            matmul,
        }
    }
}

fn get_embedding_table(device: &Device, num_embeddings: usize, embedding_dim: usize) -> TensorF32 {
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
    device.tensor_f32(num_embeddings, embedding_dim, embeddings_table)
}

impl Operator for Embedding {
    fn name(&self) -> &str {
        "Embedding"
    }

    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, Error> {
        let inputs = &[inputs[0], &self.embedding_table];
        self.matmul.forward(inputs)
    }

    fn forward_realize(&self, _inputs: &[&Tensor], _output: &Tensor) -> Result<(), Error> {
        panic!()
    }

    fn backward(&self, _inputs: &[&Tensor], _output: &Tensor) -> Result<(), Error> {
        panic!()
    }
}
