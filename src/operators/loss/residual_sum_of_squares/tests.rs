use crate::{Device, LossFunction, Tensor};

use super::ResidualSumOfSquares;

#[test]
fn derive() {
    let expected_tensor = Tensor::new(1, 8, vec![4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]);
    let actual_tensor = Tensor::new(1, 8, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let expected_derived_loss =
        Tensor::new(1, 8, vec![-6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0]);
    let loss_function = ResidualSumOfSquares::default();
    let accelerator = Device::cpu();
    let mut actual_derived_loss = Tensor::new(0, 0, vec![0.0]);
    let op_result = loss_function.derive(
        &accelerator,
        &expected_tensor,
        &actual_tensor,
        &mut actual_derived_loss,
    );
    op_result.expect("Ok");
    assert_eq!(actual_derived_loss, expected_derived_loss);
}

#[test]
fn evaluate() {
    let expected_tensor = Tensor::new(1, 8, vec![4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]);
    let actual_tensor = Tensor::new(1, 8, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let loss_function = ResidualSumOfSquares::default();
    let accelerator = Device::cpu();
    assert_eq!(
        loss_function.evaluate(&accelerator, &expected_tensor, &actual_tensor),
        Ok((4.0 - 1.0 as f32).powf(2.0) * 8.0)
    );
}
