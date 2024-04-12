use rand::Rng;

use crate::tensor::{Error, Tensor};

#[test]
fn new() {
    // Given rows and cols
    // When a matrix is built
    // Then it has the appropriate shape

    let matrix = Tensor::new(
        4,
        3,
        vec![
            0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, //
        ],
    );
    assert_eq!((matrix.rows(), matrix.cols()), (4, 3));
}

#[test]
fn multiplication_shape_compatibility() {
    // Given two matrices with incompatible shapes
    // When a matrix multiplication is done
    // Then there is an error

    let lhs = Tensor::new(
        1,
        1,
        vec![
            0.0, //
        ],
    );

    let rhs = Tensor::new(
        2,
        1,
        vec![
            0.0, //
            0.0, //
        ],
    );

    let mut result = Tensor::default();
    result.reset(lhs.rows(), rhs.cols(), 0.0);
    let error = Tensor::gemm(false, false, 1.0, &lhs, &rhs, 0.0, &mut result, false);
    assert_eq!(error, Err(Error::IncompatibleTensorShapes))
}

#[test]
fn reshape_result() {
    let mut lhs = Tensor::new(
        2,
        4,
        vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
        ],
    );

    let expected = Tensor::new(
        1,
        8,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
            7.0, 8.0, //
        ],
    );

    lhs.reshape(1, 8).expect("Ok");
    assert_eq!(lhs, expected);
}

#[test]
fn reshape_error() {
    let mut lhs = Tensor::new(
        2,
        4,
        vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
        ],
    );

    let op_result = lhs.reshape(1, 11);
    assert_eq!(op_result, Err(Error::UnsupportedOperation));
}

#[test]
fn get() {
    let tensor = Tensor::new(
        2,
        2,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
        ],
    );

    assert_eq!(tensor.get(0, 1), 2.0);
}

#[test]
fn clip() {
    // Given a tensor with 0.0 and 1.0 values
    // When it is clipped
    // Then the clipped tensor contains clipped values

    let epsilon = 1e-8;
    let tensor = Tensor::new(
        1,
        4,
        vec![
            0.0, 1.0, //
            0.5, 0.7, //
        ],
    );
    let mut clipped = Tensor::default();
    tensor.clip(0.0 + epsilon, 1.0 - epsilon, &mut clipped);

    let expected = Tensor::new(
        1,
        4,
        vec![
            0.0 + epsilon,
            1.0 - epsilon, //
            0.5,
            0.7, //
        ],
    );

    assert_eq!(clipped, expected);
}

#[test]
fn set() {
    let mut tensor = Tensor::new(
        2,
        2,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
        ],
    );

    tensor.set(1, 0, 99.0);
    assert_eq!(tensor.get(1, 0), 99.0);
}

#[test]
fn assign() {
    let mut tensor = Tensor::new(
        2,
        2,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
        ],
    );

    let tensor2 = Tensor::new(
        3,
        3,
        vec![
            11.0, 12.0, 13.0, //
            14.0, 15.0, 16.0, //
            17.0, 18.0, 19.0, //
        ],
    );

    tensor.assign(&tensor2);
    assert_eq!(tensor, tensor2);
}

#[test]
fn reset() {
    // Given a tensor
    // When it is reset to a bigger shape
    // Then all values are set to 0.0

    let mut tensor = Tensor::new(
        1,
        1,
        vec![
            1.0, //
        ],
    );
    tensor.reset(2, 1, Default::default());
    let expected = Tensor::new(
        2,
        1,
        vec![
            0.0, //
            0.0, //
        ],
    );
    assert_eq!(tensor, expected)
}

#[test]
fn matrix_multiplication_result() {
    // Given a left-hand side matrix and and a right-hand side matrix
    // When the multiplication lhs * rhs is done
    // Then the resulting matrix has the correct values

    let lhs = Tensor::new(
        3,
        2,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
        ],
    );
    let rhs = Tensor::new(
        2,
        3,
        vec![
            11.0, 12.0, 13.0, //
            14.0, 15.0, 16.0, //
        ],
    );
    let expected_result = Tensor::new(
        3,
        3,
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

    let mut result = Tensor::default();
    result.reset(lhs.rows(), rhs.cols(), 0.0);
    _ = Tensor::gemm(false, false, 1.0, &lhs, &rhs, 0.0, &mut result, false);
    assert_eq!(result, expected_result);
}

#[test]
fn transposed_lhs_matrix_multiplication_result() {
    // Given a left-hand side matrix and and a right-hand side matrix
    // When the multiplication lhs * rhs is done
    // Then the resulting matrix has the correct values

    let lhs2 = Tensor::new(
        3,
        2,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
        ],
    );
    let mut lhs = Tensor::default();
    lhs2.transpose(&mut lhs);
    let rhs = Tensor::new(
        2,
        3,
        vec![
            11.0, 12.0, 13.0, //
            14.0, 15.0, 16.0, //
        ],
    );
    let expected_result = Tensor::new(
        3,
        3,
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

    let mut result = Tensor::default();
    result.reset(lhs.cols(), rhs.cols(), 0.0);
    _ = Tensor::gemm(true, false, 1.0, &lhs, &rhs, 0.0, &mut result, false);
    assert_eq!(result, expected_result);
}

#[test]
fn transposed_rhs_matrix_multiplication_result() {
    // Given a left-hand side matrix and and a right-hand side matrix
    // When the multiplication lhs * rhs is done
    // Then the resulting matrix has the correct values

    let lhs = Tensor::new(
        3,
        2,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
        ],
    );
    let rhs2 = Tensor::new(
        2,
        3,
        vec![
            11.0, 12.0, 13.0, //
            14.0, 15.0, 16.0, //
        ],
    );
    let mut rhs = Tensor::default();
    rhs2.transpose(&mut rhs);
    let expected_result = Tensor::new(
        3,
        3,
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

    let mut result = Tensor::default();
    result.reset(lhs.rows(), rhs.rows(), 0.0);
    Tensor::gemm(false, true, 1.0, &lhs, &rhs, 0.0, &mut result, false).expect("Ok");
    assert_eq!(result, expected_result);
}

#[test]
fn lhs_t_rhs_t_result_matrix_multiplication_result() {
    let lhs2 = Tensor::new(
        3,
        2,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
        ],
    );
    let mut lhs = Tensor::default();
    lhs2.transpose(&mut lhs);
    let rhs2 = Tensor::new(
        2,
        3,
        vec![
            11.0, 12.0, 13.0, //
            14.0, 15.0, 16.0, //
        ],
    );
    let mut rhs = Tensor::default();
    rhs2.transpose(&mut rhs);
    let expected_result = Tensor::new(
        3,
        3,
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

    let mut result = Tensor::default();
    result.reset(lhs.cols(), rhs.rows(), 0.0);
    Tensor::gemm(true, true, 1.0, &lhs, &rhs, 0.0, &mut result, false).expect("Ok");
    assert_eq!(result, expected_result);
}

#[test]
fn lhs_t_rhs_t_result_t_matrix_multiplication_result() {
    let lhs2 = Tensor::new(
        4,
        2,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
            7.0, 8.0,
        ],
    );
    let mut lhs = Tensor::default();
    lhs2.transpose(&mut lhs);

    let rhs2 = Tensor::new(
        2,
        3,
        vec![
            11.0, 12.0, 13.0, //
            14.0, 15.0, 16.0, //
        ],
    );

    let mut rhs = Tensor::default();
    rhs2.transpose(&mut rhs);

    let expected_result2 = Tensor::new(
        4,
        3,
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
            7.0 * 11.0 + 8.0 * 14.0,
            7.0 * 12.0 + 8.0 * 15.0,
            7.0 * 13.0 + 8.0 * 16.0, //
        ],
    );
    let mut expected_result = Tensor::default();
    expected_result2.transpose(&mut expected_result);

    let mut result = Tensor::default();
    result.reset(lhs.cols(), rhs.rows(), 0.0);
    Tensor::gemm(true, true, 1.0, &lhs, &rhs, 0.0, &mut result, true).expect("Ok");
    assert_eq!(result, expected_result);
}

#[test]
fn lhs_t_rhs_result_t_matrix_multiplication_result() {
    let lhs2 = Tensor::new(
        3,
        2,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
        ],
    );
    let mut lhs = Tensor::default();
    lhs2.transpose(&mut lhs);
    let rhs = Tensor::new(
        2,
        3,
        vec![
            11.0, 12.0, 13.0, //
            14.0, 15.0, 16.0, //
        ],
    );

    let expected_result2 = Tensor::new(
        3,
        3,
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
    let mut expected_result = Tensor::default();
    expected_result2.transpose(&mut expected_result);

    let mut result = Tensor::default();
    result.reset(lhs.cols(), rhs.cols(), 0.0);
    Tensor::gemm(true, false, 1.0, &lhs, &rhs, 0.0, &mut result, true).expect("Ok");
    assert_eq!(result, expected_result);
}

#[test]
fn matrix_addition_result() {
    // Given a left-hand side matrix and and a right-hand side matrix
    // When the addition lhs + rhs is done
    // Then the resulting matrix has the correct values

    let lhs = Tensor::new(
        3,
        2,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
        ],
    );
    let rhs: Tensor = Tensor::new(
        3,
        2,
        vec![
            11.0, 12.0, //
            14.0, 15.0, //
            13.0, 16.0, //
        ],
    );
    let expected_result = Tensor::new(
        3,
        2,
        vec![
            1.0 + 11.0,
            2.0 + 12.0, //
            3.0 + 14.0,
            4.0 + 15.0, //
            5.0 + 13.0,
            6.0 + 16.0, //
        ],
    );

    let mut result = Tensor::default();
    _ = lhs.add(&rhs, &mut result);
    assert_eq!(result, expected_result);
}

#[test]
fn element_wise_mul_result() {
    // Given a left-hand side matrix and and a right-hand side matrix
    // When the element-wise multiplication is done
    // Then the resulting matrix has the correct values

    let lhs = Tensor::new(
        3,
        2,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
        ],
    );
    let rhs: Tensor = Tensor::new(
        3,
        2,
        vec![
            11.0, 12.0, //
            14.0, 15.0, //
            13.0, 16.0, //
        ],
    );
    let expected_result = Tensor::new(
        3,
        2,
        vec![
            1.0 * 11.0,
            2.0 * 12.0, //
            3.0 * 14.0,
            4.0 * 15.0, //
            5.0 * 13.0,
            6.0 * 16.0, //
        ],
    );

    let mut result = Tensor::default();
    _ = lhs.element_wise_mul(&rhs, &mut result);
    assert_eq!(result, expected_result);
}

#[test]
fn scalar_mul() {
    // Given a left-hand side matrix and and a right-hand scalar
    // When scalar multiplication is done
    // Then the resulting matrix has the correct values

    let lhs = Tensor::new(
        3,
        2,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
        ],
    );
    let rhs = -2.0;
    let expected_result = Tensor::new(
        3,
        2,
        vec![
            1.0 * -2.0,
            2.0 * -2.0, //
            3.0 * -2.0,
            4.0 * -2.0, //
            5.0 * -2.0,
            6.0 * -2.0, //
        ],
    );

    let mut result = Tensor::default();
    _ = lhs.scalar_mul(rhs, &mut result);
    assert_eq!(result, expected_result);
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
    let m = Tensor::new(rows, cols, values);

    let mut result = Tensor::default();
    result.reset(m.rows(), m.cols(), 0.0);
    _ = Tensor::gemm(false, false, 1.0, &m, &m, 0.0, &mut result, false);
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
    let m = Tensor::new(rows, cols, values);

    let mut result = Tensor::default();
    _ = m.add(&m, &mut result);
}

#[test]
fn transpose() {
    let matrix = Tensor::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut matrix2 = Tensor::default();
    matrix.transpose(&mut matrix2);
    for row in 0..matrix.rows() {
        for col in 0..matrix.cols() {
            assert_eq!(matrix2.get(col, row), matrix.get(row, col));
        }
    }
}
