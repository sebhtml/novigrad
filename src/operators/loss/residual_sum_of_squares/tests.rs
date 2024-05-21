use std::ops::Deref;

use crate::{BinaryOperator, Device, GenericTensor};

use super::ResidualSumOfSquares;

#[test]
fn derive() {
    let device = Device::default();
    let expected_tensor = device.tensor(
        1,
        8,
        vec![4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
        &[],
        false,
        false,
    );
    let actual_tensor = device.tensor(
        1,
        8,
        vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        &[],
        true,
        false,
    );
    let expected_derived_loss =
        device.tensor_f32(1, 8, vec![-6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0]);
    let device = Device::cpu();
    let operator = ResidualSumOfSquares::new(&device);
    let loss = operator.forward(&expected_tensor, &actual_tensor).unwrap();
    loss.forward().unwrap();
    loss.compute_gradient().unwrap();
    let actual_derived_loss: &GenericTensor = &actual_tensor.gradient().deref().borrow();
    assert_eq!(actual_derived_loss, &expected_derived_loss);
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
