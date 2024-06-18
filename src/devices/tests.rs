use more_asserts::{assert_ge, assert_le};

use crate::{new_tensor, Device, DeviceTrait};

#[test]
fn clip_min() {
    use crate::devices::DeviceTrait;
    use crate::Device;
    let device = Device::default();
    let stream = device.stream().unwrap();
    let input = new_tensor!(
        device,
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
    let output = new_tensor!(device, 2, 3, vec![0.0; 6]).unwrap();

    let min = new_tensor!(device, 1, 1, vec![0.0]).unwrap();

    let max = new_tensor!(device, 1, 1, vec![f32::INFINITY]).unwrap();

    device.clip(&min, &max, &input, &output, &stream).unwrap();

    let expected = new_tensor!(
        device,
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
    use crate::devices::DeviceTrait;
    use crate::Device;
    let device = Device::default();
    let stream = device.stream().unwrap();
    let input = new_tensor!(
        device,
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
    let output = new_tensor!(device, 2, 3, vec![0.0; 6]).unwrap();

    let min = new_tensor!(device, 1, 1, vec![f32::NEG_INFINITY]).unwrap();

    let max = new_tensor!(device, 1, 1, vec![2.0]).unwrap();

    device.clip(&min, &max, &input, &output, &stream).unwrap();

    let expected = new_tensor!(
        device,
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
    use crate::devices::DeviceTrait;
    use crate::Device;
    let device = Device::default();
    let input = new_tensor!(device, 1, 100, vec![0.3; 100]).unwrap();
    let output = new_tensor!(device, 1, 100, vec![0.0; 100]).unwrap();
    let device_stream = device.stream().unwrap();
    device.bernoulli(&input, &output, &device_stream).unwrap();

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

#[test]
fn test_copy_1() {
    let device = Device::default();
    let device_stream = device.stream().unwrap();
    let mut tensor = new_tensor!(
        device,
        3,
        3,
        vec![
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0, //
        ],
    )
    .unwrap();

    let tensor2 = new_tensor!(
        device,
        3,
        3,
        vec![
            11.0, 12.0, 13.0, //
            14.0, 15.0, 16.0, //
            17.0, 18.0, 19.0, //
        ],
    )
    .unwrap();
    device
        .copy(
            tensor2.len() as i32,
            &tensor2,
            0,
            1,
            &mut tensor,
            0,
            1,
            &device_stream,
        )
        .unwrap();
    assert_eq!(tensor, tensor2);
}

#[test]
fn test_copy_2() {
    let device = Device::default();
    let device_stream = device.stream().unwrap();
    let expected = new_tensor!(
        device,
        2,
        4,
        vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
        ],
    )
    .unwrap();

    let mut actual = new_tensor!(device, 2, 4, vec![0.0; 2 * 4]).unwrap();

    device
        .copy(
            expected.len() as i32,
            &expected,
            0,
            1,
            &mut actual,
            0,
            1,
            &device_stream,
        )
        .unwrap();
    assert_eq!(actual, expected);
}
