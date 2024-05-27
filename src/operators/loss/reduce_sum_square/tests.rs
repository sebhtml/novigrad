use std::ops::Deref;

use crate::{BinaryOperator, Device, Tensor};

use super::ReduceSumSquare;

#[test]
fn derive() {
    let device = Device::default();
    let expected_tensor = device
        .tensor_with_grad(
            1,
            8,
            vec![4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
            &[],
            false,
            false,
        )
        .unwrap();
    let actual_tensor = device
        .tensor_with_grad(
            1,
            8,
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            &[],
            true,
            false,
        )
        .unwrap();
    let expected_derived_loss = device
        .tensor(1, 8, vec![-6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0])
        .unwrap();
    let operator = ReduceSumSquare::new(&device);
    let loss = operator.forward(&expected_tensor, &actual_tensor).unwrap();
    loss.forward().unwrap();
    loss.compute_gradient().unwrap();
    let actual_derived_loss: &Tensor = &actual_tensor.gradient().deref().borrow();
    assert_eq!(actual_derived_loss, &expected_derived_loss);
}

#[test]
fn evaluate() {
    let device = Device::default();
    let expected_tensor = device
        .tensor(1, 8, vec![4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
        .unwrap();
    let actual_tensor = device
        .tensor(1, 8, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        .unwrap();
    let loss = device.tensor(1, 1, vec![0.0]).unwrap();
    ReduceSumSquare::execute(&[&expected_tensor, &actual_tensor], &[&loss]).unwrap();
    assert_eq!(
        loss.get_values().unwrap()[0],
        (4.0 - 1.0 as f32).powf(2.0) * 8.0,
    );
}
