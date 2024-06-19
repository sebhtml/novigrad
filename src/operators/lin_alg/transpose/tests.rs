use crate::{new_tensor, Device, ExecutableOperator};

use super::Transpose;

#[test]
fn transpose() {
    let device = Device::default();
    let device_stream = device.new_stream().unwrap();
    let matrix = new_tensor!(device, 3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let matrix2 = new_tensor!(device, 2, 3, vec![0.0; 6]).unwrap();
    Transpose::execute(
        &Default::default(),
        &[&matrix],
        &[&matrix2],
        &device,
        &device_stream,
    )
    .unwrap();
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
