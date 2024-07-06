use std::ops::Div;

use more_asserts::{assert_ge, assert_le};

use crate::{
    new_tensor, statistics::bernoulli::Bernoulli, stream::StreamTrait, Device, DeviceTrait,
    ExecutableOperator,
};

#[test]
fn clip_min() {
    use crate::Device;
    let device = Device::default();
    let device_stream = device.new_stream().unwrap();
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

    device
        .clip(&min, &max, &input, &output, &device_stream)
        .unwrap();

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
    device_stream.wait_for().unwrap();
    assert_eq!(expected.get_values(), output.get_values(),);
}

#[test]
fn clip_max() {
    use crate::devices::DeviceTrait;
    use crate::Device;
    let device = Device::default();
    let device_stream = device.new_stream().unwrap();
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

    device
        .clip(&min, &max, &input, &output, &device_stream)
        .unwrap();

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
    device_stream.wait_for().unwrap();
    assert_eq!(expected.get_values(), output.get_values(),);
}

#[test]
fn bernoulli() {
    use crate::Device;
    let device = Device::default();
    let probability = 0.3;
    let input = new_tensor!(device, 1, 100, vec![probability; 100]).unwrap();
    let output = new_tensor!(device, 1, 100, vec![0.0; 100]).unwrap();
    let device_stream = device.new_stream().unwrap();
    Bernoulli::execute(
        &crate::OperatorAttributes::F32(probability),
        &[&input],
        &[&output],
        &device,
        &device_stream,
    )
    .unwrap();
    device_stream.wait_for().unwrap();
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
    let device_stream = device.new_stream().unwrap();
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
    device_stream.wait_for().unwrap();
    assert_eq!(tensor, tensor2);
}

#[test]
fn test_copy_2() {
    let device = Device::default();
    let device_stream = device.new_stream().unwrap();
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
    device_stream.wait_for().unwrap();
    assert_eq!(actual, expected);
}

fn mean(elements: &[f32]) -> f32 {
    let sum = elements.iter().sum::<f32>();
    sum / elements.len() as f32
}

fn stddev(elements: &[f32], mean: f32) -> f32 {
    let sum = elements.iter().map(|x| (x - mean).powi(2)).sum::<f32>();
    sum.div(elements.len() as f32).sqrt()
}

#[test]
fn standardization_output_data_with_mean_0_and_stddev_1_for_each_neuron() {
    use crate::devices::DeviceTrait;
    use crate::Device;
    use rand_distr::{Distribution, Normal};
    let device = Device::default();
    let normal1 = Normal::new(20.0, 3.0).unwrap();
    let sample1 = |_| normal1.sample(&mut rand::thread_rng());
    let normal2 = Normal::new(40.0, 8.0).unwrap();
    let sample2 = |_| normal2.sample(&mut rand::thread_rng());
    let neurons = 2;
    let features = 16;
    let n = neurons * features;
    let input_values1 = (0..n / 2).map(sample1).collect::<Vec<_>>();
    let input_values2 = (0..n / 2).map(sample2).collect::<Vec<_>>();
    let input_values = vec![input_values1, input_values2].concat();
    let input = new_tensor!(device, neurons, features, input_values.clone()).unwrap();

    let output = new_tensor!(device, neurons, features, vec![0.0; n]).unwrap();
    let device_stream = device.new_stream().unwrap();
    device
        .standardization(&input, &output, &device_stream)
        .unwrap();
    device_stream.wait_for().unwrap();
    let elements = output.get_values().unwrap();

    let mut i = 0;
    while i < elements.len() {
        let slice = &elements[i..i + features];
        let mean = mean(slice);
        let stddev = stddev(slice, mean);
        assert_le!((mean - 0.0).abs(), 1e-5);
        assert_le!((stddev - 1.0).abs(), 1e-5);
        i += features;
    }
}

#[test]
fn standardization_output_data_with_mean_0_and_stddev_1_not_for_all_neurons() {
    use crate::devices::DeviceTrait;
    use crate::Device;
    use rand_distr::{Distribution, Normal};
    let device = Device::default();
    let normal1 = Normal::new(20.0, 3.0).unwrap();
    let sample1 = |_| normal1.sample(&mut rand::thread_rng());
    let normal2 = Normal::new(40.0, 8.0).unwrap();
    let sample2 = |_| normal2.sample(&mut rand::thread_rng());
    let neurons = 2;
    let features = 16;
    let n = neurons * features;
    let input_values1 = (0..n / 2).map(sample1).collect::<Vec<_>>();
    let input_values2 = (0..n / 2).map(sample2).collect::<Vec<_>>();
    let input_values = vec![input_values1, input_values2].concat();

    // With 2 neurons
    let input_2_neurons = new_tensor!(device, neurons, features, input_values.clone()).unwrap();
    let output_2_neurons = new_tensor!(device, neurons, features, vec![0.0; n]).unwrap();
    let device_stream = device.new_stream().unwrap();
    device
        .standardization(&input_2_neurons, &output_2_neurons, &device_stream)
        .unwrap();
    device_stream.wait_for().unwrap();
    let elements_2_neurons = output_2_neurons.get_values().unwrap();

    // Same input but with 1 neuron instead of 2 neurons.
    let input_1_neurons = new_tensor!(device, 1, n, input_values.clone()).unwrap();
    let output_1_neurons = new_tensor!(device, 1, n, vec![0.0; n]).unwrap();
    let device_stream = device.new_stream().unwrap();
    device
        .standardization(&input_1_neurons, &output_1_neurons, &device_stream)
        .unwrap();
    device_stream.wait_for().unwrap();
    let elements_1_neurons = output_1_neurons.get_values().unwrap();

    assert_eq!(n, elements_2_neurons.len());
    assert_eq!(n, elements_1_neurons.len());
    assert_ne!(elements_1_neurons, elements_2_neurons);
}

#[test]
fn standardization_is_deterministic() {
    use crate::devices::DeviceTrait;
    use crate::Device;
    use rand_distr::{Distribution, Normal};
    let device = Device::default();
    let normal1 = Normal::new(20.0, 3.0).unwrap();
    let sample1 = |_| normal1.sample(&mut rand::thread_rng());
    let normal2 = Normal::new(40.0, 8.0).unwrap();
    let sample2 = |_| normal2.sample(&mut rand::thread_rng());
    let neurons = 2;
    let features = 16;
    let n = neurons * features;
    let input_values1 = (0..n / 2).map(sample1).collect::<Vec<_>>();
    let input_values2 = (0..n / 2).map(sample2).collect::<Vec<_>>();
    let input_values = vec![input_values1, input_values2].concat();
    let input = new_tensor!(device, neurons, features, input_values.clone()).unwrap();

    let output1 = new_tensor!(device, neurons, features, vec![0.0; n]).unwrap();
    let device_stream1 = device.new_stream().unwrap();
    device
        .standardization(&input, &output1, &device_stream1)
        .unwrap();
    device_stream1.wait_for().unwrap();
    let elements1 = output1.get_values().unwrap();

    let output2 = new_tensor!(device, neurons, features, vec![0.0; n]).unwrap();
    let device_stream2 = device.new_stream().unwrap();
    device
        .standardization(&input, &output2, &device_stream2)
        .unwrap();
    device_stream2.wait_for().unwrap();
    let elements2 = output2.get_values().unwrap();

    assert_eq!(elements1, elements2);
}
