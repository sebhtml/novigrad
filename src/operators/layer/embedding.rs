use std::{cell::RefCell, ops::Deref, rc::Rc};

use crate::{accelerator::Accelerator, DeltaWorkingMemory, Error, Gradient, OperatorTrait, Tensor};
use rand::{distributions::Uniform, thread_rng, Rng};

pub struct Embedding {
    embedding_table: Rc<RefCell<Tensor>>,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        Self {
            embedding_table: Rc::new(RefCell::new(get_embedding_table(
                num_embeddings,
                embedding_dim,
            ))),
        }
    }
}

impl OperatorTrait for Embedding {
    fn compute_gradients(
        &mut self,
        accelerator: &Accelerator,
        inputs: &Vec<Rc<Tensor>>,
        layer_output_delta: &Tensor,
    ) -> Result<Vec<Gradient>, Error> {
        let mut gradients = vec![];
        let mut gradient = Tensor::default();
        let layer_input = &inputs[0];
        let a = layer_output_delta;
        let b = layer_input;
        let c = &mut gradient;
        c.reset(b.cols(), a.cols(), 0.0);
        let op_result = Tensor::matmul(accelerator, true, false, a, b, c, true);
        op_result.expect("Ok");

        gradients.push(Gradient::new(self.embedding_table.clone(), gradient));

        Ok(gradients)
    }

    fn forward(
        &mut self,
        accelerator: &Accelerator,
        inputs: &Vec<Rc<Tensor>>,
    ) -> Result<Rc<Tensor>, Error> {
        let embedding_table: &Tensor = &self.embedding_table.deref().borrow();
        debug_assert_eq!(inputs.len(), 1);
        let input = &inputs[0];
        let mut output = Tensor::default();
        debug_assert_eq!(input.cols(), embedding_table.rows());
        let a = input;
        let b = &embedding_table;
        let c = &mut output;
        c.reset(a.rows(), b.cols(), 0.0);
        Tensor::matmul(accelerator, false, false, a, b, c, false)?;
        Ok(Rc::new(output))
    }

    fn backward(
        &self,
        _inputs: &Vec<Rc<Tensor>>,
        _accelerator: &Accelerator,
        _layer_delta: &Tensor,
        _previous_layer_delta: &mut Tensor,
    ) {
        panic!("Embedding can not go backward !");
    }

    fn get_layer_output_delta(
        &self,
        accelerator: &Accelerator,
        _working_memory: &mut DeltaWorkingMemory,
        _inputs: &Vec<Rc<Tensor>>,
        _output: &Rc<Tensor>,
        back_propagated_delta: &Tensor,
        layer_delta: &mut Tensor,
    ) {
        layer_delta.assign(accelerator, back_propagated_delta)
    }

    fn name(&self) -> &str {
        "Embedding"
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
