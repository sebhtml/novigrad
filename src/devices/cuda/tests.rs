#[test]
fn cublas_sgemm_column_major() {
    use crate::devices::DeviceInterface;
    use crate::Device;

    let device = Device::cuda().unwrap();

    let (m, n, k) = (2, 4, 3);
    let a = device.tensor_f32(
        2,
        3,
        vec![
            //
            1.0, 4.0, //
            2.0, 5.0, //
            3.0, 6.0, //
        ],
    );
    let b = device.tensor_f32(
        3,
        4,
        vec![
            //
            1.0, 5.0, 9.0, //
            2.0, 6.0, 10.0, //
            3.0, 7.0, 11.0, //
            4.0, 8.0, 12.0, //
        ],
    );
    let mut c = device.tensor_f32(
        2,
        4,
        vec![
            //
            2.0, 7.0, //
            6.0, 2.0, //
            0.0, 7.0, //
            4.0, 2.0, //
        ],
    );

    device
        .sgemm(
            false,
            false,
            m,
            n,
            k,
            1.0,
            a.as_ptr(),
            m,
            b.as_ptr(),
            k,
            1.0,
            c.as_mut_ptr(),
            m,
        )
        .unwrap();

    assert_eq!(
        &c.get_values().unwrap(),
        &vec![
            //
            40.0, 90.0, //
            50.0, 100.0, //
            50.0, 120.0, //
            60.0, 130.0, //
        ]
    );
}

#[test]
fn cuda_tensor() {
    use crate::Device;
    let device = Device::cuda().unwrap();
    let tensor = device.tensor_f32(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(
        tensor.get_values().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0,]
    );
}

#[test]
fn cuda_set_value() {
    use crate::Device;
    let device = Device::cuda().unwrap();
    let mut tensor = device.tensor_f32(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    tensor.set_values(vec![10.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(
        tensor.get_values().unwrap(),
        vec![10.0, 2.0, 3.0, 4.0, 5.0, 6.0,]
    );
}
