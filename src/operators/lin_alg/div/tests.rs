use crate::{new_tensor, Device, Div, ExecutableOperator, OperatorAttributes};

#[test]
fn test_div_with_valid_inputs() {
    let device = Device::default();
    let device_stream = device.new_stream().unwrap();
    let input1 = new_tensor!(
        device,
        1,
        4,
        vec![
            4.0, 3.0, //
            0.5, 8.0, //
        ],
    )
    .unwrap();
    let input2 = new_tensor!(
        device,
        1,
        4,
        vec![
            2.0, 2.0, //
            2.0, 2.0, //
        ],
    )
    .unwrap();

    let output = new_tensor!(device, 1, 4, vec![Default::default(); 4],).unwrap();

    Div::execute(
        &OperatorAttributes::None,
        &[&input1, &input2],
        &[&output],
        &device,
        &device_stream,
    )
    .unwrap();

    assert_eq!(vec![2.0, 1.5, 0.25, 4.0], output.get_values().unwrap());
}

#[test]
fn test_div_by_0() {
    let device = Device::default();
    let device_stream = device.new_stream().unwrap();
    let input1 = new_tensor!(
        device,
        1,
        4,
        vec![
            4.0, 3.0, //
            0.5, 8.0, //
        ],
    )
    .unwrap();
    let input2 = new_tensor!(
        device,
        1,
        4,
        vec![
            0.0, 2.0, //
            2.0, 0.0, //
        ],
    )
    .unwrap();

    let output = new_tensor!(device, 1, 4, vec![Default::default(); 4],).unwrap();

    Div::execute(
        &OperatorAttributes::None,
        &[&input1, &input2],
        &[&output],
        &device,
        &device_stream,
    )
    .unwrap();

    assert_eq!(
        vec![f32::INFINITY, 1.5, 0.25, f32::INFINITY],
        output.get_values().unwrap()
    );
}
