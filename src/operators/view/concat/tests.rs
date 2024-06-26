use crate::{
    copy_slice, new_tensor, Concat, Device, ExecutableOperator, OperatorAttributes, Unconcat,
};

#[test]
fn concat() {
    let device = Device::default();

    let input_1 = new_tensor!(
        device,
        2,
        3,
        vec![
            //
            11.0, 12.0, 13.0, //
            21.0, 22.0, 23.0, //
        ],
    )
    .unwrap();

    let input_2 = new_tensor!(
        device,
        2,
        3,
        vec![
            //
            14.0, 15.0, 16.0, //
            24.0, 25.0, 26.0, //
        ],
    )
    .unwrap();

    let input_3 = new_tensor!(
        device,
        2,
        3,
        vec![
            //
            17.0, 18.0, 19.0, //
            27.0, 28.0, 29.0, //
        ],
    )
    .unwrap();

    let output = new_tensor!(device, 2, 9, vec![0.0; 2 * 9]).unwrap();

    let expected = new_tensor!(
        device,
        2,
        9,
        vec![
            //
            11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, //
            21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, //
        ],
    )
    .unwrap();

    let inputs = [&input_1, &input_2, &input_3];
    let outputs = [&output];
    let device_stream = device.new_stream().unwrap();
    Concat::execute(
        &OperatorAttributes::default(),
        &inputs,
        &outputs,
        &device,
        &device_stream,
    )
    .unwrap();
    assert_eq!(*output.size(), *expected.size());
    assert_eq!(output.get_values(), expected.get_values());
}

#[test]
fn unconcat() {
    let device = Device::default();

    let input = new_tensor!(
        device,
        2,
        9,
        vec![
            11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, //
            21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, //
        ],
    )
    .unwrap();

    let output_1 = new_tensor!(device, 2, 3, vec![0.0; 2 * 3]).unwrap();
    let output_2 = new_tensor!(device, 2, 3, vec![0.0; 2 * 3]).unwrap();
    let output_3 = new_tensor!(device, 2, 3, vec![0.0; 2 * 3]).unwrap();

    let inputs = &[&input];
    let outputs = &[&output_1, &output_2, &output_3];

    let expected_output_1 = new_tensor!(
        device,
        2,
        3,
        vec![
            //
            11.0, 12.0, 13.0, //
            21.0, 22.0, 23.0, //
        ],
    )
    .unwrap();

    let expected_output_2 = new_tensor!(
        device,
        2,
        3,
        vec![
            //
            14.0, 15.0, 16.0, //
            24.0, 25.0, 26.0, //
        ],
    )
    .unwrap();

    let expected_output_3 = new_tensor!(
        device,
        2,
        3,
        vec![
            //
            17.0, 18.0, 19.0, //
            27.0, 28.0, 29.0, //
        ],
    )
    .unwrap();

    let device_stream = device.new_stream().unwrap();
    Unconcat::execute(
        &Default::default(),
        inputs,
        outputs,
        &device,
        &device_stream,
    )
    .unwrap();

    assert_eq!(output_1.get_values(), expected_output_1.get_values());
    assert_eq!(output_2.get_values(), expected_output_2.get_values());
    assert_eq!(output_3.get_values(), expected_output_3.get_values());
}

#[test]
fn test_copy_slice() {
    let device = Device::default();
    let device_stream = device.new_stream().unwrap();
    let from = new_tensor!(
        device,
        2,
        2,
        vec![
            11.0, 12.0, //
            13.0, 14.0, //
        ],
    )
    .unwrap();

    let mut actual = new_tensor!(
        device,
        2,
        4,
        vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
        ],
    )
    .unwrap();

    let expected = new_tensor!(
        device,
        2,
        4,
        vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 11.0, 12.0, //
        ],
    )
    .unwrap();

    copy_slice(
        from.cols(),
        &from,
        0,
        0,
        &mut actual,
        1,
        2,
        &device,
        &device_stream,
    )
    .unwrap();
    assert_eq!(actual, expected);
}
