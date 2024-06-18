use rand::Rng;

use crate::{new_tensor, Add, Device, DeviceTrait, ExecutableOperator};

#[test]
fn matrix_addition_result() {
    // Given a left-hand side matrix and and a right-hand side matrix
    // When the addition lhs + rhs is done
    // Then the resulting matrix has the correct values

    let device = Device::default();
    let device_stream = device.stream().unwrap();

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
    let result = new_tensor!(device, rows, cols, vec![0.0; len]).unwrap();
    Add::execute(
        &Default::default(),
        &[&lhs, &rhs],
        &[&result],
        &device,
        &device_stream,
    )
    .unwrap();
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
    Add::execute(
        &Default::default(),
        &[&m, &m],
        &[&result],
        &device,
        &device_stream,
    )
    .unwrap();
}
