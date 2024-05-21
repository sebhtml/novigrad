use crate::Device;

use super::ResidualSumOfSquares;

#[test]
fn derive() {
    let device = Device::default();
    let expected_tensor = device.tensor_f32(1, 8, vec![4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]);
    let actual_tensor = device.tensor_f32(1, 8, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let expected_derived_loss =
        device.tensor_f32(1, 8, vec![-6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0]);
    let device = Device::cpu();
    let mut actual_derived_loss = device.tensor_f32(1, 8, vec![0.0; 8]);
    ResidualSumOfSquares::derive(&expected_tensor, &actual_tensor, &mut actual_derived_loss)
        .unwrap();
    assert_eq!(actual_derived_loss, expected_derived_loss);
}

#[test]
fn evaluate() {
    let device = Device::default();
    let expected_tensor = device.tensor_f32(1, 8, vec![4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]);
    let actual_tensor = device.tensor_f32(1, 8, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    assert_eq!(
        ResidualSumOfSquares::evaluate(&expected_tensor, &actual_tensor),
        Ok((4.0 - 1.0 as f32).powf(2.0) * 8.0)
    );
}
