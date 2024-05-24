use rand::Rng;

use crate::{Device, Gemm};

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
    let m = device.tensor(rows, cols, values).unwrap();

    let rows = m.rows();
    let cols = m.cols();
    let len = rows * cols;
    let mut result = device.tensor(rows, cols, vec![0.0; len]).unwrap();

    let alpha = 1.0;
    let beta = 1.0;
    Gemm::gemm(false, false, alpha, &m, &m, beta, &mut result, false).unwrap();
}

#[test]
fn lhs_t_rhs_result_t_matrix_multiplication_result() {
    let device = Device::default();
    let lhs2 = device
        .tensor(
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
    let mut lhs = device.tensor(2, 4, vec![0.0; 8]).unwrap();
    lhs2.transpose(&mut lhs).unwrap();

    let rhs = device
        .tensor(
            2,
            3,
            vec![
                11.0, 12.0, 13.0, //
                14.0, 15.0, 16.0, //
            ],
        )
        .unwrap();

    let expected_result2 = device
        .tensor(
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
    let mut expected_result = device.tensor(3, 4, vec![0.0; 12]).unwrap();
    expected_result2.transpose(&mut expected_result).unwrap();

    let rows = rhs.cols();
    let cols = lhs.cols();
    let len = rows * cols;
    let mut result = device.tensor(rows, cols, vec![0.0; len]).unwrap();

    let alpha = 1.0;
    let beta = 1.0;
    Gemm::gemm(true, false, alpha, &lhs, &rhs, beta, &mut result, true).unwrap();
    assert_eq!(result, expected_result);
}

#[test]
fn transposed_lhs_matrix_multiplication_result() {
    let device = Device::default();
    // Given a left-hand side matrix and and a right-hand side matrix
    // When the multiplication lhs * rhs is done
    // Then the resulting matrix has the correct values

    let lhs2 = device
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
    let mut lhs = device.tensor(2, 3, vec![0.0; 6]).unwrap();
    lhs2.transpose(&mut lhs).unwrap();
    let rhs = device
        .tensor(
            2,
            3,
            vec![
                11.0, 12.0, 13.0, //
                14.0, 15.0, 16.0, //
            ],
        )
        .unwrap();
    let expected_result = device
        .tensor(
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
    let mut result = device.tensor(rows, cols, vec![0.0; len]).unwrap();

    let alpha = 1.0;
    let beta = 1.0;
    Gemm::gemm(true, false, alpha, &lhs, &rhs, beta, &mut result, false).unwrap();
    assert_eq!(result, expected_result);
}

#[test]
fn matrix_multiplication_result() {
    let device = Device::default();
    // Given a left-hand side matrix and and a right-hand side matrix
    // When the multiplication lhs * rhs is done
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
            2,
            3,
            vec![
                11.0, 12.0, 13.0, //
                14.0, 15.0, 16.0, //
            ],
        )
        .unwrap();
    let expected_result = device
        .tensor(
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
    let mut result = device.tensor(rows, cols, vec![0.0; len]).unwrap();

    let alpha = 1.0;
    let beta = 1.0;
    Gemm::gemm(false, false, alpha, &lhs, &rhs, beta, &mut result, false).unwrap();
    assert_eq!(result, expected_result);
}

#[test]
fn transposed_rhs_matrix_multiplication_result() {
    let device = Device::default();
    // Given a left-hand side matrix and and a right-hand side matrix
    // When the multiplication lhs * rhs is done
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
    let rhs2 = device
        .tensor(
            2,
            3,
            vec![
                11.0, 12.0, 13.0, //
                14.0, 15.0, 16.0, //
            ],
        )
        .unwrap();
    let mut rhs = device.tensor(3, 2, vec![0.0; 6]).unwrap();
    rhs2.transpose(&mut rhs).unwrap();
    let expected_result = device
        .tensor(
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
    let mut result = device.tensor(rows, cols, vec![0.0; len]).unwrap();

    let alpha = 1.0;
    let beta = 1.0;
    Gemm::gemm(false, true, alpha, &lhs, &rhs, beta, &mut result, false).unwrap();
    assert_eq!(result, expected_result);
}

#[test]
fn lhs_t_rhs_t_result_matrix_multiplication_result() {
    let device = Device::default();
    let lhs2 = device
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
    let mut lhs = device.tensor(2, 3, vec![0.0; 6]).unwrap();
    lhs2.transpose(&mut lhs).unwrap();
    let rhs2 = device
        .tensor(
            2,
            3,
            vec![
                11.0, 12.0, 13.0, //
                14.0, 15.0, 16.0, //
            ],
        )
        .unwrap();
    let mut rhs = device.tensor(3, 2, vec![0.0; 6]).unwrap();
    rhs2.transpose(&mut rhs).unwrap();
    let expected_result = device
        .tensor(
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
    let mut result = device.tensor(rows, cols, vec![0.0; len]).unwrap();

    let alpha = 1.0;
    let beta = 1.0;
    Gemm::gemm(true, true, alpha, &lhs, &rhs, beta, &mut result, false).unwrap();
    assert_eq!(result, expected_result);
}

#[test]
fn lhs_t_rhs_t_result_t_matrix_multiplication_result() {
    let device = Device::default();
    let lhs2 = device
        .tensor(
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
    let mut lhs = device.tensor(2, 4, vec![0.0; 8]).unwrap();
    lhs2.transpose(&mut lhs).unwrap();

    let rhs2 = device
        .tensor(
            2,
            3,
            vec![
                11.0, 12.0, 13.0, //
                14.0, 15.0, 16.0, //
            ],
        )
        .unwrap();

    let mut rhs = device.tensor(3, 2, vec![0.0; 6]).unwrap();
    rhs2.transpose(&mut rhs).unwrap();

    let expected_result2 = device
        .tensor(
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
    let mut expected_result = device.tensor(3, 4, vec![0.0; 12]).unwrap();
    expected_result2.transpose(&mut expected_result).unwrap();

    let rows = rhs.rows();
    let cols = lhs.cols();
    let len = rows * cols;
    let mut result = device.tensor(rows, cols, vec![0.0; len]).unwrap();

    let alpha = 1.0;
    let beta = 1.0;
    Gemm::gemm(true, true, alpha, &lhs, &rhs, beta, &mut result, true).unwrap();
    assert_eq!(result, expected_result);
}
