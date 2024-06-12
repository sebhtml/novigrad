use crate::{
    new_tensor, new_tensor_with_grad, tensor::Tensor, BinaryOperator, Device, DeviceTrait,
};

use super::ReduceSumSquare;

#[test]
fn derive() {
    let device = Device::default();
    let expected_tensor = new_tensor_with_grad!(
        device,
        1,
        8,
        vec![4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
        &[],
        false,
        false,
    )
    .unwrap();
    let actual_tensor = new_tensor_with_grad!(
        device,
        1,
        8,
        vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        &[],
        true,
        false,
    )
    .unwrap();
    let expected_derived_loss = new_tensor!(
        device,
        1,
        8,
        vec![-6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0]
    )
    .unwrap();
    let operator = ReduceSumSquare::new(&device);
    let loss = operator.forward(&expected_tensor, &actual_tensor).unwrap();
    loss.forward(Default::default()).unwrap();
    loss.compute_gradient(Default::default()).unwrap();
    let actual_derived_loss: &Tensor = &actual_tensor.gradient();
    assert_eq!(actual_derived_loss, &expected_derived_loss);
}

#[test]
fn evaluate() {
    let device = Device::default();
    let expected_tensor =
        new_tensor!(device, 1, 8, vec![4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]).unwrap();
    let actual_tensor =
        new_tensor!(device, 1, 8, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
    let loss = new_tensor!(device, 1, 1, vec![0.0]).unwrap();
    device
        .reduce_square_sum(&expected_tensor, &actual_tensor, &loss)
        .unwrap();
    assert_eq!(
        loss.get_values().unwrap()[0],
        (4.0 - 1.0 as f32).powf(2.0) * 8.0,
    );
}
