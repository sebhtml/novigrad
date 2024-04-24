use std::{cell::RefCell, ops::Deref, rc::Rc};

use crate::{devices::Device, DeltaWorkingMemory, Error, Gradient, OperatorTrait, Tensor};
use rand::{distributions::Uniform, thread_rng, Rng};

pub struct Embedding {
    embedding_table: Rc<RefCell<Tensor>>,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize, device: &Device) -> Self {
        Self {
            embedding_table: Rc::new(RefCell::new(get_embedding_table(
                device,
                num_embeddings,
                embedding_dim,
            ))),
        }
    }
}

impl OperatorTrait for Embedding {
    fn backward(
        &self,
        device: &Device,
        _error_working_memory: &mut DeltaWorkingMemory,
        inputs: &Vec<Rc<RefCell<Tensor>>>,
        _output: &Rc<RefCell<Tensor>>,
        back_propagated_delta: &mut Tensor,
        layer_delta: &mut Tensor,
    ) -> Result<(Tensor, Vec<Gradient>), Error> {
        {
            layer_delta.assign(device, back_propagated_delta);
        }

        let mut gradients = vec![];
        {
            let mut gradient = device.tensor(0, 0, vec![]);
            let input: &Tensor = &inputs[0].deref().borrow();
            let a: &Tensor = layer_delta;
            let b: &Tensor = input;
            let c: &mut Tensor = &mut gradient;
            c.reset(b.cols(), a.cols(), 0.0);
            let op_result = Tensor::matmul(device, true, false, a, b, c, true);
            op_result.expect("Ok");

            gradients.push(Gradient::new(self.embedding_table.clone(), gradient));
        }

        let mut gradient = device.tensor(0, 0, vec![]);
        gradient.assign(device, layer_delta);

        Ok((gradient, gradients))
    }

    fn forward(
        &self,
        device: &Device,
        inputs: &Vec<Rc<RefCell<Tensor>>>,
    ) -> Result<Rc<RefCell<Tensor>>, Error> {
        let embedding_table: &Tensor = &self.embedding_table.deref().borrow();
        debug_assert_eq!(inputs.len(), 1);
        let input: &Tensor = &inputs[0].deref().borrow();
        let mut output = device.tensor(0, 0, vec![]);
        debug_assert_eq!(input.cols(), embedding_table.rows());
        let a = input;
        let b = &embedding_table;
        let c = &mut output;
        c.reset(a.rows(), b.cols(), 0.0);
        Tensor::matmul(device, false, false, a, b, c, false)?;
        Ok(Rc::new(RefCell::new(output)))
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
