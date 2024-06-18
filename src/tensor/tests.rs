use std::vec;

use crate::{new_tensor, tensor::ErrorEnum, Device};

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
