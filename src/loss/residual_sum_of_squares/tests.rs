use crate::{loss::LossFunction, Tensor};

use super::ResidualSumOfSquares;

#[test]
fn derive() {
    let expected_tensor = Tensor::new(1, 8, vec![4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]);
    let actual_tensor = Tensor::new(1, 8, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let expected_derived_loss =
        Tensor::new(1, 8, vec![-6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0]);
    let loss_function = ResidualSumOfSquares::default();
    let mut actual_derived_loss = Tensor::default();
    let op_result =
        loss_function.derive(&expected_tensor, &actual_tensor, &mut actual_derived_loss);
    op_result.expect("Ok");
    assert_eq!(actual_derived_loss, expected_derived_loss);
}

#[test]
fn evaluate() {
    let expected_tensor = Tensor::new(1, 8, vec![4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]);
    let actual_tensor = Tensor::new(1, 8, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let loss_function = ResidualSumOfSquares::default();
    assert_eq!(
        loss_function.evaluate(&expected_tensor, &actual_tensor),
        Ok((4.0 - 1.0 as f32).powf(2.0) * 8.0)
    );
}
