use std::{cell::RefCell, ops::Deref, rc::Rc};

use crate::{devices::Device, DeltaWorkingMemory, Error, LearningTensor, OperatorTrait, Tensor};
use rand::{distributions::Uniform, thread_rng, Rng};

pub struct Embedding {
    embedding_table: LearningTensor,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize, device: &Device) -> Self {
        let embedding_table_gradient = device.tensor(0, 0, vec![]);
        Self {
            embedding_table: LearningTensor::new(
                Rc::new(RefCell::new(get_embedding_table(
                    device,
                    num_embeddings,
                    embedding_dim,
                ))),
                Rc::new(RefCell::new(embedding_table_gradient)),
            ),
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
    ) -> Result<Vec<LearningTensor>, Error> {
        let back_propagated_delta: &Tensor = &output.gradient().deref().borrow();
        let mut enabled_gradients = vec![];
        {
            let embedding_table_gradient: &mut Tensor =
                &mut self.embedding_table.gradient().deref().borrow_mut();
            let input: &Tensor = &inputs[0].tensor().deref().borrow();
            let a: &Tensor = back_propagated_delta;
            let b: &Tensor = input;
            let c: &mut Tensor = embedding_table_gradient;
            c.reset(b.cols(), a.cols(), 0.0);
            let op_result = Tensor::matmul(device, true, false, a, b, c, true);
            op_result.expect("Ok");
        }

        enabled_gradients.push(self.embedding_table.clone());

        {
            let backward_gradient: &mut Tensor = &mut inputs[0].gradient().deref().borrow_mut();
            backward_gradient.assign(device, back_propagated_delta);
        }

        Ok(enabled_gradients)
    }

    fn forward(
        &self,
        device: &Device,
        inputs: &Vec<LearningTensor>,
    ) -> Result<LearningTensor, Error> {
        let output = device.learning_tensor(0, 0, vec![]);
        let embedding_table: &Tensor = &self.embedding_table.tensor().deref().borrow();
        debug_assert_eq!(inputs.len(), 1);
        {
            let input: &Tensor = &inputs[0].tensor().deref().borrow();
            let output: &mut Tensor = &mut output.tensor().deref().borrow_mut();
            debug_assert_eq!(input.cols(), embedding_table.rows());
            let a = input;
            let b = &embedding_table;
            let c = output;
            c.reset(a.rows(), b.cols(), 0.0);
            Tensor::matmul(device, false, false, a, b, c, false)?;
        }

        Ok(output)
    }

    fn name(&self) -> &str {
        "Embedding"
    }
}

fn get_embedding_table(device: &Device, num_embeddings: usize, embedding_dim: usize) -> Tensor {
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
    device.tensor(num_embeddings, embedding_dim, embeddings_table)
}
