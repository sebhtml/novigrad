use std::{cell::RefCell, rc::Rc};

use crate::{OperatorTrait, Tensor};

pub struct Record {
    operator: Rc<RefCell<Box<dyn OperatorTrait>>>,
    inputs: Vec<Rc<Tensor>>,
    output: Rc<Tensor>,
}

impl Record {
    pub fn new(
        operator: Rc<RefCell<Box<dyn OperatorTrait>>>,
        inputs: Vec<Rc<Tensor>>,
        output: Rc<Tensor>,
    ) -> Self {
        Self {
            operator,
            inputs,
            output,
        }
    }

    pub fn operator(&self) -> &Rc<RefCell<Box<dyn OperatorTrait>>> {
        &self.operator
    }

    pub fn inputs(&self) -> &Vec<Rc<Tensor>> {
        &self.inputs
    }

    pub fn output(&self) -> &Rc<Tensor> {
        &self.output
    }
}

pub struct Tape {
    records: Vec<Record>,
}

impl Default for Tape {
    fn default() -> Self {
        Self {
            records: Default::default(),
        }
    }
}

impl Tape {
    pub fn push(
        &mut self,
        operator: Rc<RefCell<Box<dyn OperatorTrait>>>,
        inputs: Vec<Rc<Tensor>>,
        output: Rc<Tensor>,
    ) {
        self.records.push(Record::new(operator, inputs, output))
    }

    pub fn records(&self) -> &Vec<Record> {
        &self.records
    }

    pub fn clear(&mut self) {
        self.records.clear();
    }

    pub fn print_records(&self) {
        println!("Tape records: {}", self.records.len());
        for record in self.records.iter() {
            let operator = &record.operator;
            let inputs = &record.inputs;

            let operator_name = (*operator).borrow().name().to_owned();
            println!(
                "Tape is recording a record: operator: {}  inputs: {}  output: {}",
                operator_name,
                inputs.len(),
                1
            );
        }
    }
}
