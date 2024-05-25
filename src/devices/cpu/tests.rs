/// This is the example from https://docs.rs/cblas/latest/cblas/.
#[test]
fn cblas_sgemm_column_major() {
    use crate::devices::DeviceInterface;
    use crate::Device;
    let device = Device::cpu();
    let (m, n, k) = (2, 4, 3);
    let a = device
        .tensor(
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
    let b = device
        .tensor(
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
    let c = device
        .tensor(
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

    let alpha = 1.0;
    let beta = 1.0;

    device
        .gemm(
            false,
            false,
            m,
            n,
            k,
            alpha,
            a.as_ptr(),
            m,
            b.as_ptr(),
            k,
            beta,
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
fn cblas_sgemm_with_column_major_layout_and_row_major_operands() {
    // From https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication
    use crate::devices::DeviceInterface;
    use crate::Device;

    let device = Device::cpu();

    let m = 2;
    let n = 4;
    let k = 3;

    let a = device
        .tensor(
            2,
            3,
            vec![
                //
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
            ],
        )
        .unwrap();

    let b = device
        .tensor(
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

    let c = device
        .tensor(
            2,
            4,
            vec![
                //
                2.0, 6.0, 0.0, 4.0, //
                7.0, 2.0, 7.0, 2.0, //
            ],
        )
        .unwrap();

    let alpha = 1.0;
    let beta = 1.0;
    device
        .gemm(
            false,
            false,
            n,
            m,
            k,
            alpha,
            b.as_ptr(),
            n,
            a.as_ptr(),
            k,
            beta,
            c.as_mut_ptr(),
            n,
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
    let tensor = device
        .tensor(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();
    assert_eq!(
        tensor.get_values().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0,]
    );
}

#[test]
fn cpu_set_value() {
    use crate::Device;
    let device = Device::cpu();
    let tensor = device
        .tensor(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();
    tensor
        .set_values(vec![10.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();
    assert_eq!(
        tensor.get_values().unwrap(),
        vec![10.0, 2.0, 3.0, 4.0, 5.0, 6.0,]
    );
}
