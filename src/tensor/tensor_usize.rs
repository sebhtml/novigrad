use crate::TensorTrait;

#[derive(Clone, Debug, PartialEq)]
pub struct TensorUSize {
    rows: usize,
    cols: usize,
    values: Vec<usize>,
}

impl Default for TensorUSize {
    fn default() -> Self {
        Self {
            rows: Default::default(),
            cols: Default::default(),
            values: Default::default(),
        }
    }
}

impl TensorUSize {
    pub fn new(rows: usize, cols: usize, values: Vec<usize>) -> Self {
        Self { rows, cols, values }
    }
}

impl TensorTrait<usize, TensorUSize> for TensorUSize {
    fn rows(&self) -> usize {
        panic!("Not implemented");
    }

    fn cols(&self) -> usize {
        panic!("Not implemented");
    }

    fn row(&self, _row: usize, _result: &mut TensorUSize) {
        panic!("Not implemented");
    }

    fn index(&self, _row: usize, _col: usize) -> usize {
        panic!("Not implemented");
    }

    fn shape(&self) -> (usize, usize) {
        panic!("Not implemented");
    }

    fn reset(&mut self, new_rows: usize, new_cols: usize) {
        self.rows = new_rows;
        self.cols = new_cols;
        let values = self.rows * self.cols;
        self.values.clear();
        self.values.resize(values, Default::default())
    }

    fn values<'a>(&'a self) -> &'a Vec<usize> {
        &self.values
    }

    fn get(&self, _row: usize, _col: usize) -> usize {
        panic!("Not implemented");
    }

    fn set(&mut self, _row: usize, _col: usize, _value: usize) {
        panic!("Not implemented");
    }

    fn assign(&mut self, from: &TensorUSize) {
        self.reset(from.rows, from.cols);

        let len = from.values.len();
        let mut index = 0;
        while index < len {
            self.values[index] = from.values[index];
            index += 1;
        }
    }

    fn transpose(&self, _other: &mut TensorUSize) {
        panic!("Not implemented");
    }

    fn add(&self, _right: &TensorUSize, _result: &mut TensorUSize) -> Result<(), crate::Error> {
        panic!("Not implemented");
    }

    fn sub(&self, _right: &TensorUSize, _result: &mut TensorUSize) -> Result<(), crate::Error> {
        panic!("Not implemented");
    }

    fn element_wise_mul(
        &self,
        _right: &TensorUSize,
        _result: &mut TensorUSize,
    ) -> Result<(), crate::Error> {
        panic!("Not implemented");
    }

    fn div(&self, _right: &TensorUSize, _result: &mut TensorUSize) -> Result<(), crate::Error> {
        panic!("Not implemented");
    }

    fn matmul(
        _lhs: &TensorUSize,
        _rhs: &TensorUSize,
        _result: &mut TensorUSize,
        _options: u32,
    ) -> Result<(), crate::Error> {
        panic!("Not implemented");
    }

    fn clip(&self, _min: usize, _max: usize, _result: &mut TensorUSize) {
        panic!("Not implemented");
    }

    fn scalar_add(&self, _right: usize, _result: &mut TensorUSize) -> Result<(), crate::Error> {
        panic!("Not implemented");
    }

    fn scalar_mul(&self, _right: usize, _result: &mut TensorUSize) -> Result<(), crate::Error> {
        panic!("Not implemented");
    }

    fn add_to_row(&mut self, row: usize, rhs: &TensorUSize) -> Result<(), crate::Error> {
        panic!("Not implemented");
    }
}
