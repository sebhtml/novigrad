use std::{cell::RefCell, rc::Rc};

use crate::{DifferentiableModuleEnum, Tensor};

pub struct Record {
    pub module: Rc<RefCell<DifferentiableModuleEnum>>,
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
    pub fn push(&mut self, module: &Rc<RefCell<DifferentiableModuleEnum>>, output: &Rc<Tensor>) {
        self.records.push(Record {
            module: module.clone(),
            output: output.clone(),
        })
    }

    pub fn clear(&mut self) {
        self.records.clear();
    }
}
