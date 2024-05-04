use std::{cell::RefCell, ops::Deref, rc::Rc};

use crate::{devices::Device, Error, Identity, OperatorTrait, Tensor, TensorF32};
use rand::{distributions::Uniform, thread_rng, Rng};

#[derive(Clone)]
pub struct Embedding {
    embedding_table: Tensor,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize, device: &Device) -> Self {
        let embedding_table = get_embedding_table(device, num_embeddings, embedding_dim);
        let len = embedding_table.len();
        let mut transposed = device.tensor_f32(embedding_dim, num_embeddings, vec![0.0; len]);
        // TODO don't unwrap directly
        embedding_table.transpose(&mut transposed).unwrap();
        // TODO don't unwrap directly
        let embedding_table = device.tensor(
            Rc::new(RefCell::new(Box::new(Identity::default()))),
            &vec![],
            transposed.rows(),
            transposed.cols(),
            transposed.get_values().unwrap(),
            true,
        );

        Self { embedding_table }
    }
}

impl OperatorTrait for Embedding {
    fn backward(&self, device: &Device, inputs: &[Tensor], output: &Tensor) -> Result<(), Error> {
        let output_gradient: &TensorF32 = &output.gradient().deref().borrow();
        let embedding_table_gradient: &mut TensorF32 =
            &mut self.embedding_table.gradient().deref().borrow_mut();
        let input: &TensorF32 = &inputs[0].tensor().deref().borrow();
        let a: &TensorF32 = output_gradient;
        let b: &TensorF32 = input;
        let c: &mut TensorF32 = embedding_table_gradient;
        TensorF32::gemm(device, true, false, 1.0, a, b, 1.0, c, true)
    }

    fn forward(&self, device: &Device, inputs: &[Tensor]) -> Result<Tensor, Error> {
        let input: &TensorF32 = &inputs[0].tensor().deref().borrow();
        debug_assert_eq!(inputs.len(), 1);
        let embedding_table: &TensorF32 = &self.embedding_table.tensor().deref().borrow();
        debug_assert_eq!(input.cols(), embedding_table.cols());

        let a = input;
        let b = &embedding_table;
        let rows = a.rows();
        let cols = b.rows();
        let len = rows * cols;
        let output = device.tensor(
            Rc::new(RefCell::new(Box::new(self.clone()))),
            inputs,
            rows,
            cols,
            vec![0.0; len],
            false,
        );

        {
            let c = &mut output.tensor().deref().borrow_mut();
            TensorF32::matmul(device, false, true, a, b, c, false)?;
        }

        Ok(output)
    }

    fn name(&self) -> &str {
        "Embedding"
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
