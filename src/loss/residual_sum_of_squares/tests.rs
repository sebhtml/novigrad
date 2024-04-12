use crate::{loss::LossFunction, Tensor};

use super::ResidualSumOfSquares;

#[test]
fn derive() {
    let expected_tensor = Tensor::new(8, 1, vec![4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]);
    let actual_tensor = Tensor::new(8, 1, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let expected_derived_loss =
        Tensor::new(8, 1, vec![-6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0]);
    let loss_function = ResidualSumOfSquares::default();
    let mut tmp = Tensor::default();
    let mut actual_derived_loss = Tensor::default();
    let op_result = loss_function.derive(
        &mut tmp,
        &expected_tensor,
        &actual_tensor,
        &mut actual_derived_loss,
    );
    op_result.expect("Ok");
    assert_eq!(actual_derived_loss, expected_derived_loss);
}
