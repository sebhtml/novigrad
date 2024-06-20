use crate::{
    new_tensor, reduce_l2::ReduceL2, stream::StreamTrait, ClipNorm, Device, ExecutableOperator,
    OperatorAttributes,
};

#[test]
fn normalize() {
    let device = Device::default();
    let device_stream = device.new_stream().unwrap();
    let tensor = new_tensor!(
        device,
        1,
        4,
        vec![
            0.0, 1.0, //
            0.5, 0.7, //
        ],
    )
    .unwrap();

    let expected_norm = 1.0;
    let actual_norm = new_tensor!(device, 1, 1, vec![0.0]).unwrap();

    ReduceL2::execute(
        &OperatorAttributes::None,
        &[&tensor],
        &[&actual_norm],
        &device,
        &device_stream,
    )
    .unwrap();
    device_stream.wait_for().unwrap();
    assert_ne!(actual_norm.get_values().unwrap()[0], expected_norm);

    ClipNorm::execute(
        &Default::default(),
        &[&tensor],
        &[&tensor],
        &device,
        &device_stream,
    )
    .unwrap();

    ReduceL2::execute(
        &OperatorAttributes::None,
        &[&tensor],
        &[&actual_norm],
        &device,
        &device_stream,
    )
    .unwrap();
    device_stream.wait_for().unwrap();
    assert_eq!(actual_norm.get_values().unwrap()[0], expected_norm);
}
