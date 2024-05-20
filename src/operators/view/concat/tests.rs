use crate::{Concat, ConcatBackward, Device};

#[test]
fn concat() {
    let device = Device::default();

    let input_1 = device.tensor_f32(
        2,
        3,
        vec![
            //
            11.0, 12.0, 13.0, //
            21.0, 22.0, 23.0, //
        ],
    );

    let input_2 = device.tensor_f32(
        2,
        3,
        vec![
            //
            14.0, 15.0, 16.0, //
            24.0, 25.0, 26.0, //
        ],
    );

    let input_3 = device.tensor_f32(
        2,
        3,
        vec![
            //
            17.0, 18.0, 19.0, //
            27.0, 28.0, 29.0, //
        ],
    );

    let output = device.tensor_f32(2, 9, vec![0.0; 2 * 9]);

    let expected = device.tensor_f32(
        2,
        9,
        vec![
            //
            11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, //
            21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, //
        ],
    );

    let inputs = [&input_1, &input_2, &input_3];
    let outputs = [&output];
    Concat::execute(&inputs, &outputs).unwrap();
    assert_eq!(output.size(), expected.size());
    assert_eq!(output.get_values(), expected.get_values());
}

#[test]
fn unconcat() {
    let device = Device::default();

    let input = device.tensor_f32(
        2,
        9,
        vec![
            11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, //
            21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, //
        ],
    );

    let output_1 = device.tensor_f32(2, 3, vec![0.0; 2 * 3]);
    let output_2 = device.tensor_f32(2, 3, vec![0.0; 2 * 3]);
    let output_3 = device.tensor_f32(2, 3, vec![0.0; 2 * 3]);

    let inputs = &[&input];
    let outputs = &[&output_1, &output_2, &output_3];

    let expected_output_1 = device.tensor_f32(
        2,
        3,
        vec![
            //
            11.0, 12.0, 13.0, //
            21.0, 22.0, 23.0, //
        ],
    );

    let expected_output_2 = device.tensor_f32(
        2,
        3,
        vec![
            //
            14.0, 15.0, 16.0, //
            24.0, 25.0, 26.0, //
        ],
    );

    let expected_output_3 = device.tensor_f32(
        2,
        3,
        vec![
            //
            17.0, 18.0, 19.0, //
            27.0, 28.0, 29.0, //
        ],
    );

    ConcatBackward::execute(inputs, outputs).unwrap();

    assert_eq!(output_1.get_values(), expected_output_1.get_values());
    assert_eq!(output_2.get_values(), expected_output_2.get_values());
    assert_eq!(output_3.get_values(), expected_output_3.get_values());
}
