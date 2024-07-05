use crate::{
    new_tensor, new_tensor_with_grad, stream::StreamTrait, tensor::Tensor, BinaryOperator, Device,
    ExecutableOperator,
};

use super::SumOfSquaredErrors;

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
    let operator = SumOfSquaredErrors::new(&device);
    let loss = operator.forward(&expected_tensor, &actual_tensor).unwrap();
    let device_stream = device.new_stream().unwrap();
    loss.forward(&device, &device_stream).unwrap();
    loss.compute_gradient(&device, &device_stream).unwrap();
    let actual_derived_loss: &Tensor = &actual_tensor.gradient();
    device_stream.wait_for().unwrap();
    assert_eq!(actual_derived_loss, &expected_derived_loss);
}

#[test]
fn evaluate() {
    let device = Device::default();
    let device_stream = device.new_stream().unwrap();
    let expected_tensor =
        new_tensor!(device, 1, 8, vec![4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]).unwrap();
    let actual_tensor =
        new_tensor!(device, 1, 8, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
    let loss = new_tensor!(device, 1, 1, vec![0.0]).unwrap();
    SumOfSquaredErrors::execute(
        &Default::default(),
        &[&expected_tensor, &actual_tensor],
        &[&loss],
        &device,
        &device_stream,
    )
    .unwrap();
    assert_eq!(
        loss.get_values().unwrap()[0],
        (4.0 - 1.0_f32).powf(2.0) * 8.0,
    );
}
