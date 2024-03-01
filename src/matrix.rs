pub struct Matrix {
    rows: u32,
    cols: u32,
    cells: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: u32, cols: u32, cells: Vec<f32>) -> Self {
        Self { rows, cols, cells }
    }

    pub fn shape(&self) -> (u32, u32) {
        (self.rows, self.cols)
    }
}

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;

    #[test]
    fn constructor_test() {
        // Given rows and cols
        // When a matrix is built
        // Then it has the appropriate shape

        let matrix = Matrix::new(
            4,
            3,
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        assert_eq!(matrix.shape(), (4, 3));
    }
}
