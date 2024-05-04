use crate::Tensor;

pub struct Record {
    output: Tensor,
}

impl Record {
    pub fn new(output: Tensor) -> Self {
        Self { output }
    }

    pub fn output(&self) -> &Tensor {
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
    pub fn push(&mut self, output: Tensor) {
        self.records.push(Record::new(output))
    }

    pub fn records(&self) -> &Vec<Record> {
        &self.records
    }

    pub fn clear(&mut self) {
        self.records.clear();
    }
}
