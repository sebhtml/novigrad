use crate::{Device, LossFunction};

use super::ResidualSumOfSquares;

#[test]
fn derive() {
    let device = Device::default();
    let expected_tensor = device.tensor(1, 8, vec![4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]);
    let actual_tensor = device.tensor(1, 8, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let expected_derived_loss =
        device.tensor(1, 8, vec![-6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0]);
    let loss_function = ResidualSumOfSquares::default();
    let device = Device::cpu();
    let mut actual_derived_loss = device.tensor(0, 0, vec![]);
    let op_result = loss_function.derive(
        &device,
        &expected_tensor,
        &actual_tensor,
        &mut actual_derived_loss,
    );
    op_result.expect("Ok");
    assert_eq!(actual_derived_loss, expected_derived_loss);
}

#[test]
fn evaluate() {
    let device = Device::default();
    let expected_tensor = device.tensor(1, 8, vec![4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]);
    let actual_tensor = device.tensor(1, 8, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let loss_function = ResidualSumOfSquares::default();
    let device = Device::cpu();
    assert_eq!(
        loss_function.evaluate(&device, &expected_tensor, &actual_tensor),
        Ok((4.0 - 1.0 as f32).powf(2.0) * 8.0)
    );
}
