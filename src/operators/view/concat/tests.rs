use std::{ops::Deref, rc::Rc};

use crate::{Concat, Device, Identity, NaryOperator, TensorF32};

#[test]
fn forward() {
    let device = Device::default();

    let input_1 = device.tensor(
        Rc::new(Identity::new(&device)),
        &vec![],
        2,
        3,
        vec![
            //
            11.0, 12.0, 13.0, //
            21.0, 22.0, 23.0, //
        ],
        false,
        false,
    );

    let input_2 = device.tensor(
        Rc::new(Identity::new(&device)),
        &vec![],
        2,
        3,
        vec![
            //
            14.0, 15.0, 16.0, //
            24.0, 25.0, 26.0, //
        ],
        false,
        false,
    );

    let input_3 = device.tensor(
        Rc::new(Identity::new(&device)),
        &vec![],
        2,
        3,
        vec![
            //
            17.0, 18.0, 19.0, //
            27.0, 28.0, 29.0, //
        ],
        false,
        false,
    );

    let concat = Concat::new(&device);
    let output = concat.forward(&[&input_1, &input_2, &input_3]).unwrap();
    output.forward().unwrap();
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
