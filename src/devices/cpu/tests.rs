use crate::new_tensor;

/// This is the example from https://docs.rs/cblas/latest/cblas/.
#[test]
fn cblas_sgemm_column_major() {
    use crate::devices::DeviceTrait;
    use crate::Device;
    let device = Device::cpu();
    let device_stream = device.new_stream().unwrap();
    let (m, n, k) = (2, 4, 3);
    let a = new_tensor!(
        device,
        2,
        3,
        vec![
            //
            1.0, 4.0, //
            2.0, 5.0, //
            3.0, 6.0, //
        ],
    )
    .unwrap();
    let b = new_tensor!(
        device,
        3,
        4,
        vec![
            //
            1.0, 5.0, 9.0, //
            2.0, 6.0, 10.0, //
            3.0, 7.0, 11.0, //
            4.0, 8.0, 12.0, //
        ],
    )
    .unwrap();
    let c = new_tensor!(
        device,
        2,
        4,
        vec![
            //
            2.0, 7.0, //
            6.0, 2.0, //
            0.0, 7.0, //
            4.0, 2.0, //
        ],
    )
    .unwrap();

    let alpha = new_tensor!(device, 1, 1, vec![1.0]).unwrap();
    let beta = new_tensor!(device, 1, 1, vec![1.0]).unwrap();

    device
        .gemm(
            false,
            false,
            m,
            n,
            k,
            &alpha,
            &a,
            m,
            &b,
            k,
            &beta,
            &c,
            m,
            &device_stream,
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
fn cblas_sgemm_with_column_major_layout_and_row_major_operands() {
    // From https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication
    use crate::devices::DeviceTrait;
    use crate::Device;

    let device = Device::cpu();
    let device_stream = device.new_stream().unwrap();
    let m = 2;
    let n = 4;
    let k = 3;

    let a = new_tensor!(
        device,
        2,
        3,
        vec![
            //
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
        ],
    )
    .unwrap();

    let b = new_tensor!(
        device,
        3,
        4,
        vec![
            //
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0, //
        ],
    )
    .unwrap();

    let c = new_tensor!(
        device,
        2,
        4,
        vec![
            //
            2.0, 6.0, 0.0, 4.0, //
            7.0, 2.0, 7.0, 2.0, //
        ],
    )
    .unwrap();

    let alpha = new_tensor!(device, 1, 1, vec![1.0]).unwrap();
    let beta = new_tensor!(device, 1, 1, vec![1.0]).unwrap();

    device
        .gemm(
            false,
            false,
            n,
            m,
            k,
            &alpha,
            &b,
            n,
            &a,
            k,
            &beta,
            &c,
            n,
            &device_stream,
        )
        .unwrap();

    assert_eq!(
        &c.get_values().unwrap(),
        &vec![
            //
            40.0, 50.0, 50.0, 60.0, //
            90.0, 100.0, 120.0, 130.0, //
        ]
    );
}

#[test]
fn cpu_tensor() {
    use crate::Device;
    let device = Device::cpu();
    let tensor = new_tensor!(device, 2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert_eq!(
        tensor.get_values().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0,]
    );
}
