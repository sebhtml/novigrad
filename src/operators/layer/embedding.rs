use std::ops::Deref;

use crate::{devices::Device, DeltaWorkingMemory, Error, LearningTensor, OperatorTrait, Tensor};
use rand::{distributions::Uniform, thread_rng, Rng};

pub struct Embedding {
    embedding_table: LearningTensor,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize, device: &Device) -> Self {
        Self {
            embedding_table: get_embedding_table(device, num_embeddings, embedding_dim),
        }
    }
}

impl OperatorTrait for Embedding {
    fn backward(
        &self,
        device: &Device,
        _error_working_memory: &mut DeltaWorkingMemory,
        inputs: &Vec<LearningTensor>,
        output: &LearningTensor,
    ) -> Result<(), Error> {
        let output_gradient: &Tensor = &output.gradient().deref().borrow();
        let embedding_table_gradient: &mut Tensor =
            &mut self.embedding_table.gradient().deref().borrow_mut();
        let input: &Tensor = &inputs[0].tensor().deref().borrow();
        let a: &Tensor = output_gradient;
        let b: &Tensor = input;
        let c: &mut Tensor = embedding_table_gradient;
        c.reset(b.cols(), a.cols(), 0.0)?;
        Tensor::matmul(device, true, false, a, b, c, true)
    }

    fn forward(
        &self,
        device: &Device,
        inputs: &Vec<LearningTensor>,
    ) -> Result<LearningTensor, Error> {
        let input: &Tensor = &inputs[0].tensor().deref().borrow();
        debug_assert_eq!(inputs.len(), 1);
        let embedding_table: &Tensor = &self.embedding_table.tensor().deref().borrow();
        debug_assert_eq!(input.cols(), embedding_table.rows());

        let output = device.learning_tensor(0, 0, vec![], false);

        {
            let a = input;
            let b = &embedding_table;
            let c = &mut output.tensor().deref().borrow_mut();
            c.reset(a.rows(), b.cols(), 0.0)?;
            Tensor::matmul(device, false, false, a, b, c, false)?;
        }

        Ok(output)
    }

    fn name(&self) -> &str {
        "Embedding"
    }
}

fn get_embedding_table(
    device: &Device,
    num_embeddings: usize,
    embedding_dim: usize,
) -> LearningTensor {
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
    device.learning_tensor(num_embeddings, embedding_dim, embeddings_table, true)
}
