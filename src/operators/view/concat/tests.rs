use std::{ops::Deref, rc::Rc};

use crate::{Concat, ConcatBackward, Device, Instruction, NaryOperator, TensorF32};

#[test]
fn forward() {
    let device = Device::default();

    let input_1 = device.tensor(
        2,
        3,
        vec![
            //
            11.0, 12.0, 13.0, //
            21.0, 22.0, 23.0, //
        ],
        &[],
        false,
        false,
    );

    let input_2 = device.tensor(
        2,
        3,
        vec![
            //
            14.0, 15.0, 16.0, //
            24.0, 25.0, 26.0, //
        ],
        &[],
        false,
        false,
    );

    let input_3 = device.tensor(
        2,
        3,
        vec![
            //
            17.0, 18.0, 19.0, //
            27.0, 28.0, 29.0, //
        ],
        &[],
        false,
        false,
    );

    let concat = Concat::new(&device);
    let output = concat.forward(&[&input_1, &input_2, &input_3]).unwrap();
    output.forward_instructions().deref().borrow()[0]
        .forward()
        .unwrap();
    let output: &TensorF32 = &output.tensor().deref().borrow();

    let expected = TensorF32::new(
        2,
        9,
        vec![
            //
            11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, //
            21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, //
        ],
        &device,
    );

    assert_eq!(output.size(), expected.size());
    assert_eq!(output.get_values(), expected.get_values());
}

#[test]
fn backward() {
    let device = Device::default();

    let input_1 = device.tensor(2, 3, vec![0.0; 2 * 3], &[], true, false);

    let input_2 = device.tensor(2, 3, vec![0.0; 2 * 3], &[], true, false);

    let input_3 = device.tensor(2, 3, vec![0.0; 2 * 3], &[], true, false);

    let concat_b = ConcatBackward::default();
    let output = device.tensor(2, 9, vec![0.0; 2 * 9], &[], true, false);
    let inputs = &[&input_1, &input_2, &input_3];
    let outputs = &[&output];
    let instruction = Instruction::new(Rc::new(concat_b), outputs, inputs);

    output.gradient().deref().borrow_mut().set_values(vec![
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, //
        21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, //
    ]);

    instruction.forward().unwrap();

    let expected_input_1_gradient = device.tensor_f32(
        2,
        3,
        vec![
            //
            11.0, 12.0, 13.0, //
            21.0, 22.0, 23.0, //
        ],
    );

    let expected_input_2_gradient = device.tensor_f32(
        2,
        3,
        vec![
            //
            14.0, 15.0, 16.0, //
            24.0, 25.0, 26.0, //
        ],
    );

    let expected_input_3_gradient = device.tensor_f32(
        2,
        3,
        vec![
            //
            17.0, 18.0, 19.0, //
            27.0, 28.0, 29.0, //
        ],
    );

    assert_eq!(
        input_1.gradient().deref().borrow().size(),
        expected_input_1_gradient.size()
    );

    assert_eq!(
        input_1.gradient().deref().borrow().get_values(),
        expected_input_1_gradient.get_values()
    );

    assert_eq!(
        input_2.gradient().deref().borrow().size(),
        expected_input_2_gradient.size()
    );
    assert_eq!(
        input_2.gradient().deref().borrow().get_values(),
        expected_input_2_gradient.get_values()
    );

    assert_eq!(
        input_3.gradient().deref().borrow().size(),
        expected_input_3_gradient.size()
    );
    assert_eq!(
        input_3.gradient().deref().borrow().get_values(),
        expected_input_3_gradient.get_values()
    );
}
