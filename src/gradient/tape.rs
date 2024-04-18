use std::{cell::RefCell, rc::Rc};

use crate::{OperatorEnum, Tensor};

pub struct Record {
    pub operator: Rc<RefCell<OperatorEnum>>,
    pub output: Rc<Tensor>,
}

pub struct Tape {
    pub records: Vec<Record>,
}

impl Default for Tape {
    fn default() -> Self {
        Self {
            records: Default::default(),
        }
    }
}

impl Tape {
    pub fn push(&mut self, operator: &Rc<RefCell<OperatorEnum>>, output: &Rc<Tensor>) {
        self.records.push(Record {
            operator: operator.clone(),
            output: output.clone(),
        })
    }

    pub fn clear(&mut self) {
        self.records.clear();
    }
}
