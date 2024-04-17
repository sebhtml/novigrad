use std::{cell::RefCell, rc::Rc};

use crate::{
    Accelerator, DifferentiableModule, DifferentiableModuleEnum, Embedding, Linear, Reshape,
    Softmax, Tape,
};

pub struct Session {
    accelerator: Rc<Accelerator>,
    tape: Rc<RefCell<Tape>>,
}

impl Default for Session {
    fn default() -> Self {
        Self {
            accelerator: Default::default(),
            tape: Default::default(),
        }
    }
}

impl Session {
    pub fn accelerator(&self) -> Rc<Accelerator> {
        self.accelerator.clone()
    }

    pub fn tape(&self) -> Rc<RefCell<Tape>> {
        self.tape.clone()
    }

    pub fn embedding(&self, num_embeddings: usize, embedding_dim: usize) -> DifferentiableModule {
        DifferentiableModule::new(
            self.accelerator(),
            self.tape(),
            Rc::new(RefCell::new(DifferentiableModuleEnum::Embedding(
                Embedding::new(num_embeddings, embedding_dim),
            ))),
        )
    }

    pub fn reshape(
        &self,
        input_rows: usize,
        input_cols: usize,
        output_rows: usize,
        output_cols: usize,
    ) -> DifferentiableModule {
        DifferentiableModule::new(
            self.accelerator(),
            self.tape(),
            Rc::new(RefCell::new(DifferentiableModuleEnum::Reshape(
                Reshape::new(input_rows, input_cols, output_rows, output_cols),
            ))),
        )
    }

    pub fn linear(
        &self,
        weights_rows: usize,
        weights_cols: usize,
        bias_rows: usize,
    ) -> DifferentiableModule {
        DifferentiableModule::new(
            self.accelerator(),
            self.tape(),
            Rc::new(RefCell::new(DifferentiableModuleEnum::Linear(Linear::new(
                weights_rows,
                weights_cols,
                bias_rows,
            )))),
        )
    }

    pub fn softmax(&self, using_cross_entropy_loss: bool) -> DifferentiableModule {
        DifferentiableModule::new(
            self.accelerator(),
            self.tape(),
            Rc::new(RefCell::new(DifferentiableModuleEnum::Softmax(
                Softmax::new(using_cross_entropy_loss),
            ))),
        )
    }
}
