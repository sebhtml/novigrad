use std::fmt::Display;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    dimensions: Vec<usize>,
    values: Vec<f32>,
}

impl Default for Tensor {
    fn default() -> Self {
        Self {
            dimensions: Default::default(),
            values: Default::default(),
        }
    }
}

impl Tensor {
    pub fn new(dimensions: Vec<usize>, values: Vec<f32>) -> Self {
        Self { dimensions, values }
    }

    pub fn dimensions(&self) -> Vec<usize> {
        self.dimensions.clone()
    }

    pub fn reshape(&mut self, new_dimensions: &Vec<usize>) {
        self.dimensions.resize(new_dimensions.len(), 0);
        let mut values = 1;
        let mut dimension = 0;
        let dimensions = self.dimensions.len();
        while dimension < dimensions {
            self.dimensions[dimension] = new_dimensions[dimension];
            values *= new_dimensions[dimension];
            dimension += 1;
        }
        self.values.resize(values, 0.0)
    }

    pub fn index(&self, indices: &Vec<usize>) -> usize {
        if indices.len() == 3 {
            indices[0] * self.dimensions[1] * self.dimensions[2]
                + indices[1] * self.dimensions[2]
                + indices[2]
        } else if indices.len() == 2 {
            indices[0] * self.dimensions[1] + indices[1]
        } else if indices.len() == 1 {
            indices[0]
        } else {
            usize::MAX
        }
    }

    pub fn get(&self, indices: &Vec<usize>) -> f32 {
        let index = self.index(indices);
        self.values[index]
    }

    pub fn set(&mut self, indices: &Vec<usize>, value: f32) {
        let index = self.index(indices);
        self.values[index] = value;
    }

    pub fn transpose(&self) -> Self {
        let rev_dimensions = self.dimensions.clone().into_iter().rev().collect();
        let mut other: Tensor = Tensor::new(rev_dimensions, self.values.clone());
        let mut row = 0;
        let rows = self.dimensions[0];
        let cols = self.dimensions[1];
        while row < rows {
            let mut col = 0;
            while col < cols {
                let value = self.get(&vec![row, col]);
                other.set(&vec![col, row], value);
                col += 1;
            }
            row += 1;
        }
        other
    }
}

#[derive(Debug, PartialEq)]
pub enum Error {
    IncompatibleTensorShapes,
}

fn add_matrix_tensor_and_matrix_tensor(
    left: &Tensor,
    right: &Tensor,
    result: &mut Tensor,
) -> Result<(), Error> {
    if left.dimensions != right.dimensions {
        return Err(Error::IncompatibleTensorShapes);
    }

    result.reshape(&left.dimensions);

    let result_ptr = result.values.as_mut_ptr();
    let left_ptr = left.values.as_ptr();
    let right_ptr = right.values.as_ptr();

    unsafe {
        for index in 0..left.values.len() {
            let left_cell = left_ptr.add(index);
            let right_cell = right_ptr.add(index);
            let result_cell = result_ptr.add(index);
            *result_cell = *left_cell + *right_cell;
        }
    }

    Ok(())
}

impl Tensor {
    pub fn add(&self, right: &Tensor, result: &mut Tensor) -> Result<(), Error> {
        let left = self;
        if left.dimensions.len() == 2 && right.dimensions.len() == 2 {
            add_matrix_tensor_and_matrix_tensor(left, right, result)
        } else {
            Err(Error::IncompatibleTensorShapes)
        }
    }
}

fn multiply_matrix_tensor_and_matrix_tensor(
    left: &Tensor,
    right: &Tensor,
    result: &mut Tensor,
) -> Result<(), Error> {
    if left.dimensions[1] != right.dimensions[0] {
        return Err(Error::IncompatibleTensorShapes);
    }

    result.reshape(&vec![left.dimensions[0], right.dimensions[1]]);

    let result_ptr = result.values.as_mut_ptr();
    let left_ptr = left.values.as_ptr();
    let right_ptr = right.values.as_ptr();

    let left_rows = left.dimensions[0];
    let left_cols = left.dimensions[1];
    let right_cols = right.dimensions[1];

    unsafe {
        let mut row = 0;
        while row != left_rows {
            let mut inner = 0;
            while inner != left_cols {
                let mut col = 0;
                while col != right_cols {
                    let left_cell = left_ptr.add(row * left.dimensions[1] + inner);
                    let right_cell = right_ptr.add(inner * right.dimensions[1] + col);
                    let result_cell = result_ptr.add(row * right.dimensions[1] + col);
                    *result_cell += *left_cell * *right_cell;
                    col += 1;
                }
                inner += 1;
            }
            row += 1;
        }
    }

    Ok(())
}

impl Tensor {
    pub fn mul(&self, right: &Tensor, result: &mut Tensor) -> Result<(), Error> {
        let left: &Tensor = self;
        if left.dimensions.len() == 2
            && right.dimensions.len() == 2
            && left.dimensions[1] == right.dimensions[0]
        {
            multiply_matrix_tensor_and_matrix_tensor(left, right, result)
        } else {
            Err(Error::IncompatibleTensorShapes)
        }
    }
}

impl Into<Vec<f32>> for Tensor {
    fn into(self) -> Vec<f32> {
        self.values
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        _ = write!(f, "Shape: {:?}", self.dimensions);
        _ = write!(f, "\n");
        for row in 0..self.dimensions[0] {
            for col in 0..self.dimensions[1] {
                let value = self.get(&vec![row, col]);
                if value < 0.0 {
                    _ = write!(f, " {:2.8}", value);
                } else {
                    _ = write!(f, " +{:2.8}", value);
                }
            }
            _ = write!(f, "\n");
        }
        Ok(())
    }
}
