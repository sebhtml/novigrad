use crate::{
    devices::Device,
    new_tensor, new_tensor_with_grad,
    tensor::{Error, Tensor},
    BinaryOperator, DeviceTrait, MatMul, TensorWithGrad, UnaryOperator,
};
use rand::{distributions::Uniform, thread_rng, Rng};

pub struct Embedding {
    embedding_table: TensorWithGrad,
    matmul: MatMul,
}

impl Embedding {
    pub fn new(
        device: &Device,
        num_embeddings: usize,
        embedding_dim: usize,
    ) -> Result<Self, Error> {
        let embedding_table = get_embedding_table(device, num_embeddings, embedding_dim)?;
        let len = embedding_table.len();
        let mut transposed = new_tensor!(device, embedding_dim, num_embeddings, vec![0.0; len])?;
        device.transpose(&embedding_table, &mut transposed)?;
        let embedding_table = new_tensor_with_grad!(
            device,
            transposed.rows(),
            transposed.cols(),
            transposed.get_values().unwrap(),
            &[],
            true,
            true,
        )?;

        let transb = true;
        let matmul = MatMul::new(device, transb);

        let op = Self {
            embedding_table,
            matmul,
        };
        Ok(op)
    }
}

impl UnaryOperator for Embedding {
    fn forward(&self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
        self.matmul.forward(input, &self.embedding_table)
    }
}

fn get_embedding_table(
    device: &Device,
    num_embeddings: usize,
    embedding_dim: usize,
) -> Result<Tensor, Error> {
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
    new_tensor!(device, num_embeddings, embedding_dim, embeddings_table)
}
