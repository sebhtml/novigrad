use std::vec;

use crate::{
    new_tensor,
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
