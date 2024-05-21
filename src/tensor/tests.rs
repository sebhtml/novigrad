use rand::Rng;

use crate::{tensor::GenericTensor, Device, ErrorEnum};

#[test]
fn new() {
    let device = Device::default();
    // Given rows and cols
    // When a matrix is built
    // Then it has the appropriate shape

    let matrix = device.tensor_f32(
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
fn reshape_result() {
    let device = Device::default();
    let lhs = device.tensor_f32(
        2,
        4,
        vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
        ],
    );

    let expected = device.tensor_f32(
        1,
        8,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
            7.0, 8.0, //
        ],
    );

    lhs.resize(&[1, 8]).unwrap();
    assert_eq!(lhs, expected);
}

#[test]
fn reshape_error() {
    let device = Device::default();
    let lhs = device.tensor_f32(
        2,
        4,
        vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
        ],
    );

    let op_result = lhs.resize(&[1, 11]);
    assert_eq!(
        op_result.map_err(|e| e.error),
        Err(ErrorEnum::UnsupportedOperation)
    );
}

#[test]
fn index() {
    let device = Device::default();
    let tensor = device.tensor_f32(
        2,
        2,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
        ],
    );

    let values = tensor.get_values().unwrap();
    assert_eq!(values[tensor.index(0, 1)], 2.0);
}

#[test]
fn clip() {
    let device = Device::default();
    let tensor = device.tensor_f32(
        1,
        4,
        vec![
            0.0, 1.0, //
            0.5, 0.7, //
        ],
    );

    let norm = 1.0;
    assert_ne!(tensor.l2_norm().unwrap(), norm);
    tensor.clip(norm).unwrap();
    assert_eq!(tensor.l2_norm().unwrap(), norm);
}

#[test]
fn set_values() {
    let device = Device::default();
    let tensor = device.tensor_f32(
        2,
        2,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
        ],
    );

    let mut values = tensor.get_values().unwrap();
    values[tensor.index(1, 0)] = 99.0;
    tensor.set_values(values);

    let values = tensor.get_values().unwrap();
    assert_eq!(values[tensor.index(1, 0)], 99.0);
}

#[test]
fn assign() {
    let device = Device::default();
    let mut tensor = device.tensor_f32(
        3,
        3,
        vec![
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0, //
        ],
    );

    let tensor2 = device.tensor_f32(
        3,
        3,
        vec![
            11.0, 12.0, 13.0, //
            14.0, 15.0, 16.0, //
            17.0, 18.0, 19.0, //
        ],
    );
    GenericTensor::copy(&tensor2, &mut tensor).unwrap();
    assert_eq!(tensor, tensor2);
}

#[test]
fn matrix_multiplication_result() {
    let device = Device::default();
    // Given a left-hand side matrix and and a right-hand side matrix
    // When the multiplication lhs * rhs is done
    // Then the resulting matrix has the correct values

    let lhs = device.tensor_f32(
        3,
        2,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
        ],
    );
    let rhs = device.tensor_f32(
        2,
        3,
        vec![
            11.0, 12.0, 13.0, //
            14.0, 15.0, 16.0, //
        ],
    );
    let expected_result = device.tensor_f32(
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

    let rows = lhs.rows();
    let cols = rhs.cols();
    let len = rows * cols;
    let mut result = device.tensor_f32(rows, cols, vec![0.0; len]);
    GenericTensor::gemm(false, false, 1.0, &lhs, &rhs, 1.0, &mut result, false).unwrap();
    assert_eq!(result, expected_result);
}

#[test]
fn transposed_lhs_matrix_multiplication_result() {
    let device = Device::default();
    // Given a left-hand side matrix and and a right-hand side matrix
    // When the multiplication lhs * rhs is done
    // Then the resulting matrix has the correct values

    let lhs2 = device.tensor_f32(
        3,
        2,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
        ],
    );
    let mut lhs = device.tensor_f32(2, 3, vec![0.0; 6]);
    lhs2.transpose(&mut lhs).unwrap();
    let rhs = device.tensor_f32(
        2,
        3,
        vec![
            11.0, 12.0, 13.0, //
            14.0, 15.0, 16.0, //
        ],
    );
    let expected_result = device.tensor_f32(
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

    let rows = lhs.cols();
    let cols = rhs.cols();
    let len = rows * cols;
    let mut result = device.tensor_f32(rows, cols, vec![0.0; len]);
    GenericTensor::gemm(true, false, 1.0, &lhs, &rhs, 1.0, &mut result, false).unwrap();
    assert_eq!(result, expected_result);
}

#[test]
fn transposed_rhs_matrix_multiplication_result() {
    let device = Device::default();
    // Given a left-hand side matrix and and a right-hand side matrix
    // When the multiplication lhs * rhs is done
    // Then the resulting matrix has the correct values

    let lhs = device.tensor_f32(
        3,
        2,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
        ],
    );
    let rhs2 = device.tensor_f32(
        2,
        3,
        vec![
            11.0, 12.0, 13.0, //
            14.0, 15.0, 16.0, //
        ],
    );
    let mut rhs = device.tensor_f32(3, 2, vec![0.0; 6]);
    rhs2.transpose(&mut rhs).unwrap();
    let expected_result = device.tensor_f32(
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

    let rows = lhs.rows();
    let cols = rhs.rows();
    let len = rows * cols;
    let mut result = device.tensor_f32(rows, cols, vec![0.0; len]);
    GenericTensor::gemm(false, true, 1.0, &lhs, &rhs, 1.0, &mut result, false).unwrap();
    assert_eq!(result, expected_result);
}

#[test]
fn lhs_t_rhs_t_result_matrix_multiplication_result() {
    let device = Device::default();
    let lhs2 = device.tensor_f32(
        3,
        2,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
        ],
    );
    let mut lhs = device.tensor_f32(2, 3, vec![0.0; 6]);
    lhs2.transpose(&mut lhs).unwrap();
    let rhs2 = device.tensor_f32(
        2,
        3,
        vec![
            11.0, 12.0, 13.0, //
            14.0, 15.0, 16.0, //
        ],
    );
    let mut rhs = device.tensor_f32(3, 2, vec![0.0; 6]);
    rhs2.transpose(&mut rhs).unwrap();
    let expected_result = device.tensor_f32(
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

    let rows = lhs.cols();
    let cols = rhs.rows();
    let len = rows * cols;
    let mut result = device.tensor_f32(rows, cols, vec![0.0; len]);
    GenericTensor::gemm(true, true, 1.0, &lhs, &rhs, 1.0, &mut result, false).unwrap();
    assert_eq!(result, expected_result);
}

#[test]
fn lhs_t_rhs_t_result_t_matrix_multiplication_result() {
    let device = Device::default();
    let lhs2 = device.tensor_f32(
        4,
        2,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
            7.0, 8.0,
        ],
    );
    let mut lhs = device.tensor_f32(2, 4, vec![0.0; 8]);
    lhs2.transpose(&mut lhs).unwrap();

    let rhs2 = device.tensor_f32(
        2,
        3,
        vec![
            11.0, 12.0, 13.0, //
            14.0, 15.0, 16.0, //
        ],
    );

    let mut rhs = device.tensor_f32(3, 2, vec![0.0; 6]);
    rhs2.transpose(&mut rhs).unwrap();

    let expected_result2 = device.tensor_f32(
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
    let mut expected_result = device.tensor_f32(3, 4, vec![0.0; 12]);
    expected_result2.transpose(&mut expected_result).unwrap();

    let rows = rhs.rows();
    let cols = lhs.cols();
    let len = rows * cols;
    let mut result = device.tensor_f32(rows, cols, vec![0.0; len]);
    GenericTensor::gemm(true, true, 1.0, &lhs, &rhs, 1.0, &mut result, true).unwrap();
    assert_eq!(result, expected_result);
}

#[test]
fn lhs_t_rhs_result_t_matrix_multiplication_result() {
    let device = Device::default();
    let lhs2 = device.tensor_f32(
        4,
        2,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
            7.0, 8.0, //
        ],
    );
    let mut lhs = device.tensor_f32(2, 4, vec![0.0; 8]);
    lhs2.transpose(&mut lhs).unwrap();

    let rhs = device.tensor_f32(
        2,
        3,
        vec![
            11.0, 12.0, 13.0, //
            14.0, 15.0, 16.0, //
        ],
    );

    let expected_result2 = device.tensor_f32(
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
    let mut expected_result = device.tensor_f32(3, 4, vec![0.0; 12]);
    expected_result2.transpose(&mut expected_result).unwrap();

    let rows = rhs.cols();
    let cols = lhs.cols();
    let len = rows * cols;
    let mut result = device.tensor_f32(rows, cols, vec![0.0; len]);
    GenericTensor::gemm(true, false, 1.0, &lhs, &rhs, 1.0, &mut result, true).unwrap();
    assert_eq!(result, expected_result);
}

#[test]
fn matrix_addition_result() {
    let device = Device::default();
    // Given a left-hand side matrix and and a right-hand side matrix
    // When the addition lhs + rhs is done
    // Then the resulting matrix has the correct values

    let lhs = device.tensor_f32(
        3,
        2,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
        ],
    );
    let rhs: GenericTensor = device.tensor_f32(
        3,
        2,
        vec![
            11.0, 12.0, //
            14.0, 15.0, //
            13.0, 16.0, //
        ],
    );
    let expected_result = device.tensor_f32(
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

    let rows = rhs.rows();
    let cols = rhs.cols();
    let len = rows * cols;
    let mut result = device.tensor_f32(rows, cols, vec![0.0; len]);
    GenericTensor::copy(&rhs, &mut result).unwrap();
    GenericTensor::add(&lhs, &mut result).unwrap();
    assert_eq!(result, expected_result);
}

#[test]
fn element_wise_mul_result() {
    let device = &Device::cpu();
    // Given a left-hand side matrix and and a right-hand side matrix
    // When the element-wise multiplication is done
    // Then the resulting matrix has the correct values

    let lhs = device.tensor_f32(
        3,
        2,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
        ],
    );
    let rhs: GenericTensor = device.tensor_f32(
        3,
        2,
        vec![
            11.0, 12.0, //
            14.0, 15.0, //
            13.0, 16.0, //
        ],
    );
    let expected_result = device.tensor_f32(
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

    let mut result = device.tensor_f32(3, 2, vec![0.0; 6]);
    GenericTensor::mul(&lhs, &rhs, &mut result).unwrap();
    assert_eq!(result, expected_result);
}

#[test]
fn scalar_mul() {
    let device = Device::default();
    // Given a left-hand side matrix and and a right-hand scalar
    // When scalar multiplication is done
    // Then the resulting matrix has the correct values

    let lhs = device.tensor_f32(
        3,
        2,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
        ],
    );
    let rhs = -2.0;
    let expected_result = device.tensor_f32(
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

    let mut result = device.tensor_f32(3, 2, vec![0.0; 6]);
    GenericTensor::copy(&lhs, &mut result).unwrap();
    GenericTensor::scalar_mul(rhs, &mut result).unwrap();
    assert_eq!(result, expected_result);
}

#[test]
fn big_matrix_multiplication() {
    let device = Device::default();
    let rows = 1024;
    let cols = 1024;
    let len = rows * cols;
    let mut values = vec![0.0; len];
    for index in 0..values.len() {
        values[index] = rand::thread_rng().gen_range(0.0..1.0)
    }
    let m = device.tensor_f32(rows, cols, values);

    let rows = m.rows();
    let cols = m.cols();
    let len = rows * cols;
    let mut result = device.tensor_f32(rows, cols, vec![0.0; len]);
    GenericTensor::gemm(false, false, 1.0, &m, &m, 1.0, &mut result, false).unwrap();
}

#[test]
fn big_matrix_addition() {
    let device = Device::default();
    let rows = 1024;
    let cols = 1024;
    let len = rows * cols;
    let mut values = vec![0.0; len];
    for index in 0..values.len() {
        values[index] = rand::thread_rng().gen_range(0.0..1.0)
    }
    let m = device.tensor_f32(rows, cols, values);

    let mut result = device.tensor_f32(rows, cols, vec![0.0; rows * cols]);
    GenericTensor::copy(&m, &mut result).unwrap();
    GenericTensor::add(&m, &mut result).unwrap();
}

#[test]
fn transpose() {
    let device = Device::default();
    let matrix = device.tensor_f32(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut matrix2 = device.tensor_f32(2, 3, vec![0.0; 6]);
    matrix.transpose(&mut matrix2).unwrap();
    let matrix_values = matrix.get_values().unwrap();
    let matrix2_values = matrix2.get_values().unwrap();
    for row in 0..matrix.rows() {
        for col in 0..matrix.cols() {
            assert_eq!(
                matrix2_values[matrix2.index(col, row)],
                matrix_values[matrix.index(row, col)]
            );
        }
    }
}

#[test]
fn copy() {
    let device = Device::default();
    let expected = device.tensor_f32(
        2,
        4,
        vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
        ],
    );

    let mut actual = device.tensor_f32(2, 4, vec![0.0; 2 * 4]);

    GenericTensor::copy(&expected, &mut actual).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn copy_slice() {
    let device = Device::default();
    let from = device.tensor_f32(
        2,
        2,
        vec![
            11.0, 12.0, //
            13.0, 14.0, //
        ],
    );

    let mut actual = device.tensor_f32(
        2,
        4,
        vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
        ],
    );

    let expected = device.tensor_f32(
        2,
        4,
        vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 11.0, 12.0, //
        ],
    );

    GenericTensor::copy_slice(from.cols(), &from, 0, 0, &mut actual, 1, 2).unwrap();
    assert_eq!(actual, expected);
}
