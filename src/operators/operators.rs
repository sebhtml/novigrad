use std::{cell::RefCell, rc::Rc};

use crate::{
    Accelerator, CrossEntropyLoss, Embedding, Linear, Operator, Reshape, ResidualSumOfSquares,
    Sigmoid, Softmax, Tape,
};

pub struct Operators {
    accelerator: Rc<Accelerator>,
    tape: Rc<RefCell<Tape>>,
}

impl Default for Operators {
    fn default() -> Self {
        Self {
            accelerator: Default::default(),
            tape: Default::default(),
        }
    }
}

impl Operators {
    pub fn accelerator(&self) -> Rc<Accelerator> {
        self.accelerator.clone()
    }

    pub fn tape(&self) -> Rc<RefCell<Tape>> {
        self.tape.clone()
    }

    pub fn embedding(&self, num_embeddings: usize, embedding_dim: usize) -> Operator {
        Operator::new(
            self.accelerator(),
            self.tape(),
            Rc::new(RefCell::new(Box::new(Embedding::new(
                num_embeddings,
                embedding_dim,
            )))),
        )
    }

    pub fn reshape(
        &self,
        input_rows: usize,
        input_cols: usize,
        output_rows: usize,
        output_cols: usize,
    ) -> Operator {
        Operator::new(
            self.accelerator(),
            self.tape(),
            Rc::new(RefCell::new(Box::new(Reshape::new(
                input_rows,
                input_cols,
                output_rows,
                output_cols,
            )))),
        )
    }

    pub fn linear(&self, weights_rows: usize, weights_cols: usize, bias_rows: usize) -> Operator {
        Operator::new(
            self.accelerator(),
            self.tape(),
            Rc::new(RefCell::new(Box::new(Linear::new(
                weights_rows,
                weights_cols,
                bias_rows,
            )))),
        )
    }

    pub fn sigmoid(&self) -> Operator {
        Operator::new(
            self.accelerator(),
            self.tape(),
            Rc::new(RefCell::new(Box::new(Sigmoid::default()))),
        )
    }

    pub fn softmax(&self, using_cross_entropy_loss: bool) -> Operator {
        Operator::new(
            self.accelerator(),
            self.tape(),
            Rc::new(RefCell::new(Box::new(Softmax::new(
                using_cross_entropy_loss,
            )))),
        )
    }

    pub fn residual_sum_of_squares(&self) -> Operator {
        Operator::new(
            self.accelerator(),
            self.tape(),
            Rc::new(RefCell::new(Box::new(ResidualSumOfSquares::default()))),
        )
    }

    pub fn cross_entropy_loss(&self) -> Operator {
        Operator::new(
            self.accelerator(),
            self.tape(),
            Rc::new(RefCell::new(Box::new(CrossEntropyLoss::default()))),
        )
    }
}
