use crate::{new_tensor, Device, DeviceTrait, ExecutableOperator, Mul};

#[test]
fn element_wise_mul_result() {
    // Given a left-hand side matrix and and a right-hand side matrix
    // When the element-wise multiplication is done
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
    let rhs = new_tensor!(
        device,
        3,
        2,
        vec![
            11.0, 12.0, //
            14.0, 15.0, //
            13.0, 16.0, //
        ],
    )
    .unwrap();
    let expected_result = new_tensor!(
        device,
        3,
        2,
        vec![
            1.0 * 11.0,
            2.0 * 12.0, //
            3.0 * 14.0,
            4.0 * 15.0, //
            5.0 * 13.0,
            6.0 * 16.0, //
        ],
    )
    .unwrap();

    let result = new_tensor!(device, 3, 2, vec![0.0; 6]).unwrap();
    Mul::execute(
        &Default::default(),
        &[&lhs, &rhs],
        &[&result],
        &device,
        &device_stream,
    )
    .unwrap();
    assert_eq!(result, expected_result);
}
