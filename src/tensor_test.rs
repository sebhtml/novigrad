#[cfg(test)]
mod tests {
    use rand::Rng;

    use crate::tensor::{Error, Tensor};

    #[test]
    fn new() {
        // Given rows and cols
        // When a matrix is built
        // Then it has the appropriate shape

        let matrix = Tensor::new(
            vec![4, 3],
            vec![
                0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, //
            ],
        );
        assert_eq!(matrix.dimensions(), vec![4, 3]);
    }

    #[test]
    fn multiplication_shape_compatibility() {
        // Given two matrices with incompatible shapes
        // When a matrix multiplication is done
        // Then there is an error

        let lhs = Tensor::new(
            vec![1, 1],
            vec![
                0.0, //
            ],
        );

        let rhs = Tensor::new(
            vec![2, 1],
            vec![
                0.0, //
                0.0, //
            ],
        );
        let actual_product = &lhs * &rhs;
        assert_eq!(actual_product, Err(Error::IncompatibleTensorShapes))
    }

    #[test]
    fn matrix_multiplication_result() {
        // Given a left-hand side matrix and and a right-hand side matrix
        // When the multiplication lhs * rhs is done
        // Then the resulting matrix has the correct values

        let lhs = Tensor::new(
            vec![3, 2],
            vec![
                1.0, 2.0, //
                3.0, 4.0, //
                5.0, 6.0, //
            ],
        );
        let rhs = Tensor::new(
            vec![2, 3],
            vec![
                11.0, 12.0, 13.0, //
                14.0, 15.0, 16.0, //
            ],
        );
        let actual_result = &lhs * &rhs;
        let expected_result = Tensor::new(
            vec![3, 3],
            vec![
                1.0 * 11.0 + 2.0 * 14.0,
                1.0 * 12.0 + 2.0 * 15.0,
                1.0 * 13.0 + 2.0 * 16.0, //
                3.0 * 11.0 + 4.0 * 14.0,
                3.0 * 12.0 + 4.0 * 15.0,
                3.0 * 13.0 + 4.0 * 16.0, //
                5.0 * 11.0 + 6.0 * 14.0,
                5.0 * 12.0 + 6.0 * 15.0,
                5.0 * 13.0 + 6.0 * 16.0, //
            ],
        );

        assert_eq!(actual_result, Ok(expected_result));
    }

    #[test]
    fn matrix_addition_result() {
        // Given a left-hand side matrix and and a right-hand side matrix
        // When the addition lhs + rhs is done
        // Then the resulting matrix has the correct values

        let lhs = Tensor::new(
            vec![3, 2],
            vec![
                1.0, 2.0, //
                3.0, 4.0, //
                5.0, 6.0, //
            ],
        );
        let rhs: Tensor = Tensor::new(
            vec![3, 2],
            vec![
                11.0, 12.0, //
                14.0, 15.0, //
                13.0, 16.0, //
            ],
        );
        let actual_result = &lhs + &rhs;
        let expected_result = Tensor::new(
            vec![3, 2],
            vec![
                1.0 + 11.0,
                2.0 + 12.0, //
                3.0 + 14.0,
                4.0 + 15.0, //
                5.0 + 13.0,
                6.0 + 16.0, //
            ],
        );

        assert_eq!(actual_result, Ok(expected_result));
    }

    #[test]
    fn big_matrix_multiplication() {
        let rows = 1024;
        let cols = 1024;
        let mut values = Vec::new();
        values.resize(rows * cols, 0.0);
        for index in 0..values.len() {
            values[index] = rand::thread_rng().gen_range(0.0..1.0)
        }
        let m = Tensor::new(vec![rows, cols], values);
        let _result = &m * &m;
    }

    #[test]
    fn big_matrix_addition() {
        let rows = 1024;
        let cols = 1024;
        let mut values = Vec::new();
        values.resize(rows * cols, 0.0);
        for index in 0..values.len() {
            values[index] = rand::thread_rng().gen_range(0.0..1.0)
        }
        let m = Tensor::new(vec![rows, cols], values);
        let _result = &m + &m;
    }

    #[test]
    fn transpose() {
        let matrix = Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let matrix2 = matrix.transpose();
        for row in 0..matrix.dimensions()[0] {
            for col in 0..matrix.dimensions()[1] {
                assert_eq!(matrix2.get(&vec![col, row]), matrix.get(&vec![row, col]));
            }
        }
    }

    #[test]
    fn add_matrix_tensor_and_vector_tensor() {
        let tensor1 = Tensor::new(
            vec![4, 3],
            vec![
                //
                0.0, 0.0, 0.0, //
                10.0, 10.0, 10.0, //
                20.0, 20.0, 20.0, //
                30.0, 30.0, 30.0, //
            ],
        );
        let tensor2 = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]);
        let result = &tensor1 + &tensor2;
        let expected = Tensor::new(
            vec![4, 3],
            vec![
                //
                1.0, 2.0, 3.0, //
                11.0, 12.0, 13.0, //
                21.0, 22.0, 23.0, //
                31.0, 32.0, 33.0, //
            ],
        );
        assert_eq!(result, Ok(expected));
    }

    #[test]
    fn multiply_vector_tensor_and_vector_tensor() {
        let tensor1 = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]);
        let tensor2 = Tensor::new(vec![1], vec![2.0]);
        assert_eq!(
            &tensor1 * &tensor2,
            Ok(Tensor::new(vec![3], vec![2.0, 4.0, 6.0,],))
        );
    }

    #[test]
    fn multiply_matrix_tensor_and_vector_tensor() {
        let tensor1 = Tensor::new(
            vec![3, 3],
            vec![
                //
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
                7.0, 8.0, 9.0, //
            ],
        );
        let tensor2 = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]);
        let expected = Tensor::new(
            vec![3, 3],
            vec![
                //
                1.0, 4.0, 9.0, //
                4.0, 10.0, 18.0, //
                7.0, 16.0, 27.0, //
            ],
        );
        assert_eq!(&tensor1 * &tensor2, Ok(expected));
    }

    #[test]
    fn multiply_matrix_tensor_and_column_matrix_tensor() {
        let tensor1 = Tensor::new(
            vec![3, 3],
            vec![
                //
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
                7.0, 8.0, 9.0, //
            ],
        );
        let tensor2 = Tensor::new(
            vec![3, 1],
            vec![
                //
                1.0, //
                2.0, //
                3.0, //
            ],
        );
        let expected = Tensor::new(
            vec![3, 3],
            vec![
                //
                1.0, 2.0, 3.0, //
                8.0, 10.0, 12.0, //
                21.0, 24.0, 27.0, //
            ],
        );
        assert_eq!(&tensor1 * &tensor2, Ok(expected));
    }

    #[test]
    fn multiply_tensor_and_column_vector() {
        let tensor1 = Tensor::new(
            vec![2, 3, 3],
            vec![
                //
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
                7.0, 8.0, 9.0, //
                //
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
                7.0, 8.0, 9.0, //
            ],
        );
        let tensor2 = Tensor::new(
            vec![3, 1],
            vec![
                //
                1.0, //
                2.0, //
                3.0, //
            ],
        );
        let expected = Tensor::new(
            vec![2, 3, 3],
            vec![
                //
                1.0, 2.0, 3.0, //
                8.0, 10.0, 12.0, //
                21.0, 24.0, 27.0, //
                //
                1.0, 2.0, 3.0, //
                8.0, 10.0, 12.0, //
                21.0, 24.0, 27.0, //
            ],
        );
        assert_eq!(&tensor1 * &tensor2, Ok(expected));
    }

    #[test]
    fn multiply_tensor_and_matrix() {
        let tensor1 = Tensor::new(
            vec![2, 3, 3],
            vec![
                //
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
                7.0, 8.0, 9.0, //
                //
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
                7.0, 8.0, 9.0, //
            ],
        );
        let tensor2 = Tensor::new(
            vec![3, 3],
            vec![
                //
                1.0, 2.0, 3.0, //
                1.0, 2.0, 3.0, //
                1.0, 2.0, 3.0, //
            ],
        );
        let expected = Tensor::new(
            vec![2, 3, 3],
            vec![
                //
                1.0, 4.0, 9.0, //
                4.0, 10.0, 18.0, //
                7.0, 16.0, 27.0, //
                //
                1.0, 4.0, 9.0, //
                4.0, 10.0, 18.0, //
                7.0, 16.0, 27.0, //
            ],
        );
        assert_eq!(&tensor1 * &tensor2, Ok(expected));
    }

    #[test]
    fn multiply_tensor_and_matrix_reduction() {
        let tensor1 = Tensor::new(
            vec![2, 3, 3],
            vec![
                //
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
                7.0, 8.0, 9.0, //
                //
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
                7.0, 8.0, 9.0, //
            ],
        );
        let tensor2 = Tensor::new(
            vec![3, 2],
            vec![
                //
                1.0, 2.0, //
                1.0, 2.0, //
                1.0, 2.0, //
            ],
        );
        let expected = Tensor::new(
            vec![2, 3, 2],
            vec![
                //
                6.0, 12.0, //
                15.0, 30.0, //
                24.0, 48.0, //
                //
                6.0, 12.0, //
                15.0, 30.0, //
                24.0, 48.0, //
            ],
        );
        assert_eq!(&tensor1 * &tensor2, Ok(expected));
    }
}
