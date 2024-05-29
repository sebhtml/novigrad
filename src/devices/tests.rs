use more_asserts::{assert_ge, assert_le};

#[test]
fn clip_min() {
    use crate::devices::DeviceInterface;
    use crate::Device;
    let device = Device::default();
    let input = device
        .tensor(
            2,
            3,
            vec![
                //
                -1.0, 4.0, //
                2.0, 5.0, //
                3.0, -6.0, //
            ],
        )
        .unwrap();
    let output = device.tensor(2, 3, vec![0.0; 6]).unwrap();

    let min = device.tensor(1, 1, vec![0.0]).unwrap();

    let max = device.tensor(1, 1, vec![f32::INFINITY]).unwrap();

    device.clip(&min, &max, &input, &output).unwrap();

    let expected = device
        .tensor(
            2,
            3,
            vec![
                //
                0.0, 4.0, //
                2.0, 5.0, //
                3.0, 0.0, //
            ],
        )
        .unwrap();
    assert_eq!(expected.get_values(), output.get_values(),);
}

#[test]
fn clip_max() {
    use crate::devices::DeviceInterface;
    use crate::Device;
    let device = Device::default();
    let input = device
        .tensor(
            2,
            3,
            vec![
                //
                -1.0, 4.0, //
                2.0, 5.0, //
                3.0, -6.0, //
            ],
        )
        .unwrap();
    let output = device.tensor(2, 3, vec![0.0; 6]).unwrap();

    let min = device.tensor(1, 1, vec![f32::NEG_INFINITY]).unwrap();

    let max = device.tensor(1, 1, vec![2.0]).unwrap();

    device.clip(&min, &max, &input, &output).unwrap();

    let expected = device
        .tensor(
            2,
            3,
            vec![
                //
                -1.0, 2.0, //
                2.0, 2.0, //
                2.0, -6.0, //
            ],
        )
        .unwrap();
    assert_eq!(expected.get_values(), output.get_values(),);
}

#[test]
fn bernoulli() {
    use crate::devices::DeviceInterface;
    use crate::Device;
    let device = Device::default();
    let input = device.tensor(1, 100, vec![0.3; 100]).unwrap();
    let output = device.tensor(1, 100, vec![0.0; 100]).unwrap();

    device.bernoulli(&input, &output).unwrap();

    let values = output.get_values().unwrap();
    let ones = values.iter().filter(|x| **x == 1.0).count();
    let zeroes = values.iter().filter(|x| **x == 0.0).count();
    assert_eq!(100, ones + zeroes);
    let diff = 10;
    assert_ge!(30 + diff, ones);
    assert_le!(30 - diff, ones);
    assert_ge!(70 + diff, zeroes);
    assert_le!(70 - diff, zeroes);
}
