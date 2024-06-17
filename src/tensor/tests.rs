use std::vec;

use rand::Rng;

use crate::{
    new_tensor,
    reduce_l2::ReduceL2,
    tensor::{ErrorEnum, Tensor},
    Device, DeviceTrait,
};

#[test]
fn new() {
    let device = Device::default();
    // Given rows and cols
    // When a matrix is built
    // Then it has the appropriate shape

    let matrix = new_tensor!(
        &device,
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
    let lhs = new_tensor!(
        device,
        2,
        4,
        vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
        ],
    )
    .unwrap();

    let expected = new_tensor!(
        device,
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
    let lhs = new_tensor!(
        device,
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
    let tensor = new_tensor!(
        device,
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
    let device_stream = device.stream().unwrap();
    let tensor = new_tensor!(
        device,
        1,
        4,
        vec![
            0.0, 1.0, //
            0.5, 0.7, //
        ],
    )
    .unwrap();

    let expected_norm = 1.0;
    let actual_norm = new_tensor!(device, 1, 1, vec![0.0]).unwrap();

    ReduceL2::execute(&[&tensor], &[&actual_norm], &device_stream).unwrap();
    assert_ne!(actual_norm.get_values().unwrap()[0], expected_norm);

    tensor.clip_norm(&device_stream).unwrap();

    ReduceL2::execute(&[&tensor], &[&actual_norm], &device_stream).unwrap();
    assert_eq!(actual_norm.get_values().unwrap()[0], expected_norm);
}

#[test]
fn set_values() {
    let device = Device::default();
    let tensor = new_tensor!(
        device,
        1,
        4,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
        ],
    )
    .unwrap();

    tensor.set_values(vec![4.0, 3.0, 2.0, 1.0]).unwrap();

    let values = tensor.get_values().unwrap();
    assert_eq!(vec![4.0, 3.0, 2.0, 1.0], values);
}

#[test]
fn assign() {
    let device = Device::default();
    let device_stream = device.stream().unwrap();
    let mut tensor = new_tensor!(
        device,
        3,
        3,
        vec![
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0, //
        ],
    )
    .unwrap();

    let tensor2 = new_tensor!(
        device,
        3,
        3,
        vec![
            11.0, 12.0, 13.0, //
            14.0, 15.0, 16.0, //
            17.0, 18.0, 19.0, //
        ],
    )
    .unwrap();
    Tensor::copy(&tensor2, &mut tensor, &device_stream).unwrap();
    assert_eq!(tensor, tensor2);
}

#[test]
fn matrix_addition_result() {
    let device = Device::default();
    let device_stream = device.stream().unwrap();
    // Given a left-hand side matrix and and a right-hand side matrix
    // When the addition lhs + rhs is done
    // Then the resulting matrix has the correct values

    let lhs = new_tensor!(
        device,
        3,
        2,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
        ],
    )
    .unwrap();
    let rhs = new_tensor!(
        device,
        3,
        2,
        vec![
            11.0, 12.0, //
            14.0, 15.0, //
            13.0, 16.0, //
        ],
    )
    .unwrap();
    let expected_result = new_tensor!(
        device,
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
    let mut result = new_tensor!(device, rows, cols, vec![0.0; len]).unwrap();
    Tensor::copy(&rhs, &mut result, &device_stream).unwrap();
    Tensor::add(&lhs, &mut result, &device_stream).unwrap();
    assert_eq!(result, expected_result);
}

#[test]
fn element_wise_mul_result() {
    let device = &Device::default();
    // Given a left-hand side matrix and and a right-hand side matrix
    // When the element-wise multiplication is done
    // Then the resulting matrix has the correct values

    let lhs = new_tensor!(
        device,
        3,
        2,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
        ],
    )
    .unwrap();
    let rhs = new_tensor!(
        device,
        3,
        2,
        vec![
            11.0, 12.0, //
            14.0, 15.0, //
            13.0, 16.0, //
        ],
    )
    .unwrap();
    let expected_result = new_tensor!(
        device,
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

    let mut result = new_tensor!(device, 3, 2, vec![0.0; 6]).unwrap();
    Tensor::mul(&lhs, &rhs, &mut result).unwrap();
    assert_eq!(result, expected_result);
}

#[test]
fn scalar_mul() {
    let device = Device::default();
    let device_stream = device.stream().unwrap();
    // Given a left-hand side matrix and and a right-hand scalar
    // When scalar multiplication is done
    // Then the resulting matrix has the correct values

    let lhs = new_tensor!(
        device,
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
    let expected_result = new_tensor!(
        device,
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

    let mut result = new_tensor!(device, 3, 2, vec![0.0; 6]).unwrap();
    Tensor::copy(&lhs, &mut result, &device_stream).unwrap();
    let rhs = new_tensor!(device, 1, 1, vec![rhs]).unwrap();
    device.scalar_mul(&rhs, &mut result).unwrap();
    assert_eq!(result, expected_result);
}

#[test]
fn big_matrix_addition() {
    let device = Device::default();
    let device_stream = device.stream().unwrap();
    let rows = 1024;
    let cols = 1024;
    let len = rows * cols;
    let mut values = vec![0.0; len];
    for index in 0..values.len() {
        values[index] = rand::thread_rng().gen_range(0.0..1.0)
    }
    let m = new_tensor!(device, rows, cols, values).unwrap();

    let result = new_tensor!(device, rows, cols, vec![0.0; rows * cols]).unwrap();
    Tensor::copy(&m, &result, &device_stream).unwrap();
    Tensor::add(&m, &result, &device_stream).unwrap();
}

#[test]
fn transpose() {
    let device = Device::default();
    let matrix = new_tensor!(device, 3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let matrix2 = new_tensor!(device, 2, 3, vec![0.0; 6]).unwrap();
    device.transpose(&matrix, &matrix2).unwrap();
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
    let device_stream = device.stream().unwrap();
    let expected = new_tensor!(
        device,
        2,
        4,
        vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
        ],
    )
    .unwrap();

    let mut actual = new_tensor!(device, 2, 4, vec![0.0; 2 * 4]).unwrap();

    Tensor::copy(&expected, &mut actual, &device_stream).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn copy_slice() {
    let device = Device::default();
    let device_stream = device.stream().unwrap();
    let from = new_tensor!(
        device,
        2,
        2,
        vec![
            11.0, 12.0, //
            13.0, 14.0, //
        ],
    )
    .unwrap();

    let mut actual = new_tensor!(
        device,
        2,
        4,
        vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
        ],
    )
    .unwrap();

    let expected = new_tensor!(
        device,
        2,
        4,
        vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 11.0, 12.0, //
        ],
    )
    .unwrap();

    Tensor::copy_slice(from.cols(), &from, 0, 0, &mut actual, 1, 2, &device_stream).unwrap();
    assert_eq!(actual, expected);
}
