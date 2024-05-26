use std::vec;

use rand::Rng;

use crate::{tensor::Tensor, Device, DeviceInterface, ErrorEnum};

#[test]
fn new() {
    let device = Device::default();
    // Given rows and cols
    // When a matrix is built
    // Then it has the appropriate shape

    let matrix = device
        .tensor(
            4,
            3,
            vec![
                0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, //
            ],
        )
        .unwrap();
    assert_eq!((matrix.rows(), matrix.cols()), (4, 3));
}

#[test]
fn resize_result() {
    let device = Device::default();
    let lhs = device
        .tensor(
            2,
            4,
            vec![
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
            ],
        )
        .unwrap();

    let expected = device
        .tensor(
            1,
            8,
            vec![
                1.0, 2.0, //
                3.0, 4.0, //
                5.0, 6.0, //
                7.0, 8.0, //
            ],
        )
        .unwrap();

    lhs.resize(&[1, 8]).unwrap();
    assert_eq!(lhs, expected);
}

#[test]
fn reshape_error() {
    let device = Device::default();
    let lhs = device
        .tensor(
            2,
            4,
            vec![
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
            ],
        )
        .unwrap();

    let op_result = lhs.resize(&[1, 11]);
    assert_eq!(
        op_result.map_err(|e| e.error),
        Err(ErrorEnum::UnsupportedOperation)
    );
}

#[test]
fn index() {
    let device = Device::default();
    let tensor = device
        .tensor(
            2,
            2,
            vec![
                1.0, 2.0, //
                3.0, 4.0, //
            ],
        )
        .unwrap();

    let values = tensor.get_values().unwrap();
    assert_eq!(values[tensor.index(0, 1)], 2.0);
}

#[test]
fn normalize() {
    let device = Device::default();
    let tensor = device
        .tensor(
            1,
            4,
            vec![
                0.0, 1.0, //
                0.5, 0.7, //
            ],
        )
        .unwrap();

    let expected_norm = 1.0;
    let actual_norm = device.tensor(1, 1, vec![0.0]).unwrap();

    tensor.l2_norm(&actual_norm).unwrap();
    assert_ne!(actual_norm.get_values().unwrap()[0], expected_norm);

    tensor.normalize().unwrap();

    tensor.l2_norm(&actual_norm).unwrap();
    assert_eq!(actual_norm.get_values().unwrap()[0], expected_norm);
}

#[test]
fn set_values() {
    let device = Device::default();
    let tensor = device
        .tensor(
            2,
            2,
            vec![
                1.0, 2.0, //
                3.0, 4.0, //
            ],
        )
        .unwrap();

    let mut values = tensor.get_values().unwrap();
    values[tensor.index(1, 0)] = 99.0;
    tensor.set_values(values).unwrap();

    let values = tensor.get_values().unwrap();
    assert_eq!(values[tensor.index(1, 0)], 99.0);
}

#[test]
fn assign() {
    let device = Device::default();
    let mut tensor = device
        .tensor(
            3,
            3,
            vec![
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
                7.0, 8.0, 9.0, //
            ],
        )
        .unwrap();

    let tensor2 = device
        .tensor(
            3,
            3,
            vec![
                11.0, 12.0, 13.0, //
                14.0, 15.0, 16.0, //
                17.0, 18.0, 19.0, //
            ],
        )
        .unwrap();
    Tensor::copy(&tensor2, &mut tensor).unwrap();
    assert_eq!(tensor, tensor2);
}

#[test]
fn matrix_addition_result() {
    let device = Device::default();
    // Given a left-hand side matrix and and a right-hand side matrix
    // When the addition lhs + rhs is done
    // Then the resulting matrix has the correct values

    let lhs = device
        .tensor(
            3,
            2,
            vec![
                1.0, 2.0, //
                3.0, 4.0, //
                5.0, 6.0, //
            ],
        )
        .unwrap();
    let rhs = device
        .tensor(
            3,
            2,
            vec![
                11.0, 12.0, //
                14.0, 15.0, //
                13.0, 16.0, //
            ],
        )
        .unwrap();
    let expected_result = device
        .tensor(
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
        )
        .unwrap();

    let rows = rhs.rows();
    let cols = rhs.cols();
    let len = rows * cols;
    let mut result = device.tensor(rows, cols, vec![0.0; len]).unwrap();
    Tensor::copy(&rhs, &mut result).unwrap();
    Tensor::add(&lhs, &mut result).unwrap();
    assert_eq!(result, expected_result);
}

#[test]
fn element_wise_mul_result() {
    let device = &Device::default();
    // Given a left-hand side matrix and and a right-hand side matrix
    // When the element-wise multiplication is done
    // Then the resulting matrix has the correct values

    let lhs = device
        .tensor(
            3,
            2,
            vec![
                1.0, 2.0, //
                3.0, 4.0, //
                5.0, 6.0, //
            ],
        )
        .unwrap();
    let rhs = device
        .tensor(
            3,
            2,
            vec![
                11.0, 12.0, //
                14.0, 15.0, //
                13.0, 16.0, //
            ],
        )
        .unwrap();
    let expected_result = device
        .tensor(
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
        )
        .unwrap();

    let mut result = device.tensor(3, 2, vec![0.0; 6]).unwrap();
    Tensor::mul(&lhs, &rhs, &mut result).unwrap();
    assert_eq!(result, expected_result);
}

#[test]
fn scalar_mul() {
    let device = Device::default();
    // Given a left-hand side matrix and and a right-hand scalar
    // When scalar multiplication is done
    // Then the resulting matrix has the correct values

    let lhs = device
        .tensor(
            3,
            2,
            vec![
                1.0, 2.0, //
                3.0, 4.0, //
                5.0, 6.0, //
            ],
        )
        .unwrap();
    let rhs = -2.0;
    let expected_result = device
        .tensor(
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
        )
        .unwrap();

    let mut result = device.tensor(3, 2, vec![0.0; 6]).unwrap();
    Tensor::copy(&lhs, &mut result).unwrap();
    let rhs = device.tensor(1, 1, vec![rhs]).unwrap();
    device.scalar_mul(&rhs, &mut result).unwrap();
    assert_eq!(result, expected_result);
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
    let m = device.tensor(rows, cols, values).unwrap();

    let result = device.tensor(rows, cols, vec![0.0; rows * cols]).unwrap();
    Tensor::copy(&m, &result).unwrap();
    Tensor::add(&m, &result).unwrap();
}

#[test]
fn transpose() {
    let device = Device::default();
    let matrix = device
        .tensor(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();
    let mut matrix2 = device.tensor(2, 3, vec![0.0; 6]).unwrap();
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
    let expected = device
        .tensor(
            2,
            4,
            vec![
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
            ],
        )
        .unwrap();

    let mut actual = device.tensor(2, 4, vec![0.0; 2 * 4]).unwrap();

    Tensor::copy(&expected, &mut actual).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn copy_slice() {
    let device = Device::default();
    let from = device
        .tensor(
            2,
            2,
            vec![
                11.0, 12.0, //
                13.0, 14.0, //
            ],
        )
        .unwrap();

    let mut actual = device
        .tensor(
            2,
            4,
            vec![
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
            ],
        )
        .unwrap();

    let expected = device
        .tensor(
            2,
            4,
            vec![
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 11.0, 12.0, //
            ],
        )
        .unwrap();

    Tensor::copy_slice(from.cols(), &from, 0, 0, &mut actual, 1, 2).unwrap();
    assert_eq!(actual, expected);
}
