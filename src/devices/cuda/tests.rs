// TODO remove ignore
#[ignore]
#[test]
fn cublas_sgemm_column_major() {
    use crate::devices::DeviceInterface;
    use crate::CudaDevice;
    use crate::Tensor;

    let device = CudaDevice::try_default().unwrap();

    let (m, n, k) = (2, 4, 3);
    let a = Tensor::new(
        2,
        3,
        vec![
            //
            1.0, 4.0, //
            2.0, 5.0, //
            3.0, 6.0, //
        ],
    );
    let b = Tensor::new(
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
    let mut c = Tensor::new(
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

    device.sgemm(false, false, m, n, k, 1.0, &a, m, &b, k, 1.0, &mut c, m);

    assert_eq!(
        c.values(),
        &vec![
            //
            40.0, 90.0, //
            50.0, 100.0, //
            50.0, 120.0, //
            60.0, 130.0, //
        ]
    );
}
