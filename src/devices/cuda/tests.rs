#[test]
fn cublas_sgemm_column_major() {
    use crate::devices::DeviceInterface;
    use crate::Device;

    let device = Device::cuda().unwrap();

    let (m, n, k) = (2, 4, 3);
    let a = device.tensor(
        2,
        3,
        vec![
            //
            1.0, 4.0, //
            2.0, 5.0, //
            3.0, 6.0, //
        ],
    );
    let b = device.tensor(
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
    let c = device.tensor(
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

    let alpha = device.tensor(1, 1, vec![1.0]);
    let beta = device.tensor(1, 1, vec![1.0]);
    device
        .gemm(false, false, m, n, k, &alpha, &a, m, &b, k, &beta, &c, m)
        .unwrap();

    let values = c.get_values().unwrap();

    assert_eq!(
        &values,
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
    let tensor = device.tensor(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(
        tensor.get_values().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0,]
    );
}

#[test]
fn cuda_set_value() {
    use crate::Device;
    let device = Device::cuda().unwrap();
    let tensor = device.tensor(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    tensor.set_values(vec![10.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(
        tensor.get_values().unwrap(),
        vec![10.0, 2.0, 3.0, 4.0, 5.0, 6.0,]
    );
}

#[test]
fn buffer() {
    use crate::Device;
    use std::ptr::null;
    let device = Device::cuda().unwrap();
    let buffer = device.buffer(32);
    assert_ne!(buffer.as_ptr(), null());
}

#[test]
fn dtoh_sync_copy_into() {
    use cudarc::driver::CudaSlice;
    let len = 32;
    let dev = cudarc::driver::CudaDevice::new(0).unwrap();
    let device_slice: CudaSlice<f32> = dev.alloc_zeros(len).unwrap();
    let mut host_slice = vec![0.0; len];
    dev.dtoh_sync_copy_into(&device_slice, &mut host_slice)
        .unwrap();
}

#[test]
fn htod_sync_copy_into() {
    use cudarc::driver::CudaSlice;
    let len = 32;
    let dev = cudarc::driver::CudaDevice::new(0).unwrap();
    let mut device_slice: CudaSlice<f32> = dev.alloc_zeros(len).unwrap();
    let values = vec![1.0; len];
    dev.htod_sync_copy_into(&values, &mut device_slice).unwrap();
    let mut host_slice = vec![0.0; len];
    dev.dtoh_sync_copy_into(&device_slice, &mut host_slice)
        .unwrap();
    assert_eq!(host_slice, values);
}

#[test]
fn sum_kernel() {
    use crate::CudaDev;
    use cudarc::driver::{LaunchAsync, LaunchConfig};
    let cuda_device = CudaDev::try_default().unwrap();
    let dev = cuda_device.dev;

    // allocate buffers
    let inp = dev.htod_copy(vec![3.0_f32; 100]).unwrap();
    let mut out = dev.alloc_zeros::<f32>(1).unwrap();

    let sum_kernel = dev.get_func("sum_kernel_module", "sum_kernel").unwrap();
    let cfg = LaunchConfig::for_num_elems(100);
    unsafe { sum_kernel.launch(cfg, (&inp, 100_usize, &mut out)) }.unwrap();

    let out_host: Vec<f32> = dev.dtoh_sync_copy(&out).unwrap();
    assert_eq!(out_host, vec![300.0],);
}

/// Example from https://github.com/coreylowman/cudarc
#[test]
fn sin_kernel() {
    use crate::CudaDev;
    use cudarc::driver::{LaunchAsync, LaunchConfig};
    let cuda_device = CudaDev::try_default().unwrap();
    let dev = cuda_device.dev;

    // allocate buffers
    let inp = dev.htod_copy(vec![1.0_f32; 100]).unwrap();
    let mut out = dev.alloc_zeros::<f32>(100).unwrap();

    let sin_kernel = dev.get_func("sin_kernel_module", "sin_kernel").unwrap();
    let cfg = LaunchConfig::for_num_elems(100);
    unsafe { sin_kernel.launch(cfg, (&mut out, &inp, 100_usize)) }.unwrap();

    let out_host: Vec<f32> = dev.dtoh_sync_copy(&out).unwrap();
    // See:
    // sin: Lack of precision?
    // https://forums.developer.nvidia.com/t/sin-lack-of-precision/14242/1
    let precision = 10e-7;
    assert_eq!(
        out_host,
        [1.0; 100]
            .map(f32::sin)
            .map(|x| ((x / precision).round()) * precision)
    );
}
