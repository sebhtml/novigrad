use crate::{new_tensor, Device, DeviceTrait, ExecutableOperator, ScalarMul};

#[test]
fn scalar_mul() {
    // Given a left-hand side matrix and and a right-hand scalar
    // When scalar multiplication is done
    // Then the resulting matrix has the correct values

    let device = Device::default();
    let device_stream = device.stream().unwrap();

    let lhs = new_tensor!(
        device,
        3,
        2,
        vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
        ],
    )
    .unwrap();
    let rhs = -2.0;
    let expected_result = new_tensor!(
        device,
        3,
        2,
        vec![
            1.0 * -2.0,
            2.0 * -2.0, //
            3.0 * -2.0,
            4.0 * -2.0, //
            5.0 * -2.0,
            6.0 * -2.0, //
        ],
    )
    .unwrap();

    let result = new_tensor!(device, 3, 2, vec![0.0; 6]).unwrap();
    let rhs = new_tensor!(device, 1, 1, vec![rhs]).unwrap();
    ScalarMul::execute(
        &Default::default(),
        &[&rhs, &lhs],
        &[&result],
        &device,
        &device_stream,
    )
    .unwrap();
    assert_eq!(result, expected_result);
}
