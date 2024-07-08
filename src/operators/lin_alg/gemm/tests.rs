use rand::Rng;

use crate::{
    new_tensor, stream::StreamTrait, transpose::Transpose, Device, ExecutableOperator, Gemm,
};

#[test]
fn big_matrix_multiplication() {
    let device = Device::default();
    let device_stream = device.new_stream().unwrap();
    let rows = 1024;
    let cols = 1024;
    let len = rows * cols;
    let mut values = vec![0.0; len];
    for index in 0..values.len() {
        values[index] = rand::thread_rng().gen_range(0.0..1.0)
    }
    let m = new_tensor!(device, rows, cols, values).unwrap();

    let rows = m.rows();
    let cols = m.cols();
    let len = rows * cols;
    let mut result = new_tensor!(device, rows, cols, vec![0.0; len]).unwrap();

    let alpha = new_tensor!(device, 1, 1, vec![1.0]).unwrap();
    let beta = new_tensor!(device, 1, 1, vec![1.0]).unwrap();

    Gemm::gemm(
        false,
        false,
        &alpha,
        &m,
        &m,
        &beta,
        &mut result,
        false,
        &device,
        &device_stream,
    )
    .unwrap();
    device_stream.wait_for().unwrap();
}

#[test]
fn lhs_t_rhs_result_t_matrix_multiplication_result() {
    let device = Device::default();
    let device_stream = device.new_stream().unwrap();
    let lhs2 = new_tensor!(
        device,
        4,
        2,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
            7.0, 8.0, //
        ],
    )
    .unwrap();
    let lhs = new_tensor!(device, 2, 4, vec![0.0; 8]).unwrap();
    Transpose::execute(
        &Default::default(),
        &[&lhs2],
        &[&lhs],
        &device,
        &device_stream,
    )
    .unwrap();

    let rhs = new_tensor!(
        device,
        2,
        3,
        vec![
            11.0, 12.0, 13.0, //
            14.0, 15.0, 16.0, //
        ],
    )
    .unwrap();

    let expected_result2 = new_tensor!(
        device,
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
    )
    .unwrap();
    let expected_result = new_tensor!(device, 3, 4, vec![0.0; 12]).unwrap();
    Transpose::execute(
        &Default::default(),
        &[&expected_result2],
        &[&expected_result],
        &device,
        &device_stream,
    )
    .unwrap();
    let rows = rhs.cols();
    let cols = lhs.cols();
    let len = rows * cols;
    let mut result = new_tensor!(device, rows, cols, vec![0.0; len]).unwrap();

    let alpha = new_tensor!(device, 1, 1, vec![1.0]).unwrap();
    let beta = new_tensor!(device, 1, 1, vec![1.0]).unwrap();

    Gemm::gemm(
        true,
        false,
        &alpha,
        &lhs,
        &rhs,
        &beta,
        &mut result,
        true,
        &device,
        &device_stream,
    )
    .unwrap();
    device_stream.wait_for().unwrap();
    assert_eq!(result, expected_result);
}

#[test]
fn transposed_lhs_matrix_multiplication_result() {
    let device = Device::default();
    let device_stream = device.new_stream().unwrap();
    // Given a left-hand side matrix and and a right-hand side matrix
    // When the multiplication lhs * rhs is done
    // Then the resulting matrix has the correct values

    let lhs2 = new_tensor!(
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
    let lhs = new_tensor!(device, 2, 3, vec![0.0; 6]).unwrap();
    Transpose::execute(
        &Default::default(),
        &[&lhs2],
        &[&lhs],
        &device,
        &device_stream,
    )
    .unwrap();
    let rhs = new_tensor!(
        device,
        2,
        3,
        vec![
            11.0, 12.0, 13.0, //
            14.0, 15.0, 16.0, //
        ],
    )
    .unwrap();
    let expected_result = new_tensor!(
        device,
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
    )
    .unwrap();

    let rows = lhs.cols();
    let cols = rhs.cols();
    let len = rows * cols;
    let mut result = new_tensor!(device, rows, cols, vec![0.0; len]).unwrap();

    let alpha = new_tensor!(device, 1, 1, vec![1.0]).unwrap();
    let beta = new_tensor!(device, 1, 1, vec![1.0]).unwrap();

    Gemm::gemm(
        true,
        false,
        &alpha,
        &lhs,
        &rhs,
        &beta,
        &mut result,
        false,
        &device,
        &device_stream,
    )
    .unwrap();
    device_stream.wait_for().unwrap();
    assert_eq!(result, expected_result);
}

#[test]
fn matrix_multiplication_result() {
    let device = Device::default();
    let device_stream = device.new_stream().unwrap();
    // Given a left-hand side matrix and and a right-hand side matrix
    // When the multiplication lhs * rhs is done
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
        2,
        3,
        vec![
            11.0, 12.0, 13.0, //
            14.0, 15.0, 16.0, //
        ],
    )
    .unwrap();
    let expected_result = new_tensor!(
        device,
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
    )
    .unwrap();

    let rows = lhs.rows();
    let cols = rhs.cols();
    let len = rows * cols;
    let mut result = new_tensor!(device, rows, cols, vec![0.0; len]).unwrap();

    let alpha = new_tensor!(device, 1, 1, vec![1.0]).unwrap();
    let beta = new_tensor!(device, 1, 1, vec![1.0]).unwrap();

    Gemm::gemm(
        false,
        false,
        &alpha,
        &lhs,
        &rhs,
        &beta,
        &mut result,
        false,
        &device,
        &device_stream,
    )
    .unwrap();
    device_stream.wait_for().unwrap();
    assert_eq!(result, expected_result);
}

#[test]
fn transposed_rhs_matrix_multiplication_result() {
    let device = Device::default();
    let device_stream = device.new_stream().unwrap();
    // Given a left-hand side matrix and and a right-hand side matrix
    // When the multiplication lhs * rhs is done
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
    let rhs2 = new_tensor!(
        device,
        2,
        3,
        vec![
            11.0, 12.0, 13.0, //
            14.0, 15.0, 16.0, //
        ],
    )
    .unwrap();
    let rhs = new_tensor!(device, 3, 2, vec![0.0; 6]).unwrap();
    Transpose::execute(
        &Default::default(),
        &[&rhs2],
        &[&rhs],
        &device,
        &device_stream,
    )
    .unwrap();
    let expected_result = new_tensor!(
        device,
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
    )
    .unwrap();

    let rows = lhs.rows();
    let cols = rhs.rows();
    let len = rows * cols;
    let mut result = new_tensor!(device, rows, cols, vec![0.0; len]).unwrap();

    let alpha = new_tensor!(device, 1, 1, vec![1.0]).unwrap();
    let beta = new_tensor!(device, 1, 1, vec![1.0]).unwrap();

    Gemm::gemm(
        false,
        true,
        &alpha,
        &lhs,
        &rhs,
        &beta,
        &mut result,
        false,
        &device,
        &device_stream,
    )
    .unwrap();
    device_stream.wait_for().unwrap();
    assert_eq!(result, expected_result);
}

#[test]
fn lhs_t_rhs_t_result_matrix_multiplication_result() {
    let device = Device::default();
    let device_stream = device.new_stream().unwrap();
    let lhs2 = new_tensor!(
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
    let lhs = new_tensor!(device, 2, 3, vec![0.0; 6]).unwrap();
    Transpose::execute(
        &Default::default(),
        &[&lhs2],
        &[&lhs],
        &device,
        &device_stream,
    )
    .unwrap();
    let rhs2 = new_tensor!(
        device,
        2,
        3,
        vec![
            11.0, 12.0, 13.0, //
            14.0, 15.0, 16.0, //
        ],
    )
    .unwrap();
    let rhs = new_tensor!(device, 3, 2, vec![0.0; 6]).unwrap();
    Transpose::execute(
        &Default::default(),
        &[&rhs2],
        &[&rhs],
        &device,
        &device_stream,
    )
    .unwrap();
    let expected_result = new_tensor!(
        device,
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
    )
    .unwrap();

    let rows = lhs.cols();
    let cols = rhs.rows();
    let len = rows * cols;
    let mut result = new_tensor!(device, rows, cols, vec![0.0; len]).unwrap();

    let alpha = new_tensor!(device, 1, 1, vec![1.0]).unwrap();
    let beta = new_tensor!(device, 1, 1, vec![1.0]).unwrap();

    Gemm::gemm(
        true,
        true,
        &alpha,
        &lhs,
        &rhs,
        &beta,
        &mut result,
        false,
        &device,
        &device_stream,
    )
    .unwrap();
    device_stream.wait_for().unwrap();
    assert_eq!(result, expected_result);
}

#[test]
fn lhs_t_rhs_t_result_t_matrix_multiplication_result() {
    let device = Device::default();
    let device_stream = device.new_stream().unwrap();
    let lhs2 = new_tensor!(
        device,
        4,
        2,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
            7.0, 8.0,
        ],
    )
    .unwrap();
    let lhs = new_tensor!(device, 2, 4, vec![0.0; 8]).unwrap();
    Transpose::execute(
        &Default::default(),
        &[&lhs2],
        &[&lhs],
        &device,
        &device_stream,
    )
    .unwrap();
    let rhs2 = new_tensor!(
        device,
        2,
        3,
        vec![
            11.0, 12.0, 13.0, //
            14.0, 15.0, 16.0, //
        ],
    )
    .unwrap();

    let rhs = new_tensor!(device, 3, 2, vec![0.0; 6]).unwrap();
    Transpose::execute(
        &Default::default(),
        &[&rhs2],
        &[&rhs],
        &device,
        &device_stream,
    )
    .unwrap();

    let expected_result2 = new_tensor!(
        device,
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
    )
    .unwrap();
    let expected_result = new_tensor!(device, 3, 4, vec![0.0; 12]).unwrap();
    Transpose::execute(
        &Default::default(),
        &[&expected_result2],
        &[&expected_result],
        &device,
        &device_stream,
    )
    .unwrap();

    let rows = rhs.rows();
    let cols = lhs.cols();
    let len = rows * cols;
    let mut result = new_tensor!(device, rows, cols, vec![0.0; len]).unwrap();

    let alpha = new_tensor!(device, 1, 1, vec![1.0]).unwrap();
    let beta = new_tensor!(device, 1, 1, vec![1.0]).unwrap();

    Gemm::gemm(
        true,
        true,
        &alpha,
        &lhs,
        &rhs,
        &beta,
        &mut result,
        true,
        &device,
        &device_stream,
    )
    .unwrap();
    device_stream.wait_for().unwrap();
    assert_eq!(result, expected_result);
}
