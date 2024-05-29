use more_asserts::assert_ge;

use crate::EPSILON;

#[test]
fn cublas_sgemm_column_major() {
    use crate::devices::DeviceInterface;
    use crate::Device;

    let device = Device::cuda().unwrap();

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
    let tensor = device
        .tensor(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();
    assert_eq!(
        tensor.get_values().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0,]
    );
}

#[test]
fn buffer() {
    use crate::slice::DevSliceTrait;
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
    let expected = [1.0; 100].map(f32::sin).into_iter().collect::<Vec<_>>();
    let expected = expected
        .into_iter()
        .map(|x| ((x / precision).round()) * precision)
        .collect::<Vec<_>>();

    assert_eq!(expected, out_host,);
}

#[test]
fn cuda_softmax_kernel() {
    use crate::CpuDevice;
    use crate::CudaDev;
    use cudarc::driver::{LaunchAsync, LaunchConfig};

    let cuda_device = CudaDev::try_default().unwrap();
    let dev = cuda_device.dev;

    let rows = 2;
    let cols = 3;
    let inp = dev
        .htod_copy(vec![
            //
            0.2 as f32, 0.3, 0.4, //
            0.4, 0.2, 0.3, //
        ])
        .unwrap();
    let mut gpu_out = dev.alloc_zeros::<f32>(6).unwrap();

    let kernel = dev
        .get_func("softmax_kernel_module", "softmax_kernel")
        .unwrap();
    let cfg = LaunchConfig::for_num_elems(100);
    unsafe { kernel.launch(cfg, (&inp, &mut gpu_out, rows, cols)) }.unwrap();

    let out_host: Vec<f32> = dev.dtoh_sync_copy(&gpu_out).unwrap();

    let mut cpu_output = vec![0.0; 6];

    {
        let input_host: Vec<f32> = dev.dtoh_sync_copy(&inp).unwrap();
        let input_host = input_host.as_ptr();
        CpuDevice::_softmax(rows, cols, input_host, cpu_output.as_mut_ptr()).unwrap();
    }

    let precision = 10e-7;
    let expected = cpu_output
        .into_iter()
        .map(|x| ((x / precision).round()) * precision)
        .collect::<Vec<_>>();

    let actual = out_host
        .into_iter()
        .map(|x| ((x / precision).round()) * precision)
        .collect::<Vec<_>>();

    assert_eq!(expected, actual,);
}

#[test]
fn cuda_dot_kernel() {
    use crate::CudaDev;
    use crate::{Device, DeviceInterface};
    use cudarc::driver::{LaunchAsync, LaunchConfig};

    let cuda_device = CudaDev::try_default().unwrap();
    let dev = cuda_device.dev;

    let n = 6;
    let left_data = vec![
        //
        0.2 as f32, 0.4, 0.6, //
        0.8, 0.2, 0.4, //
    ];
    let right_data = vec![
        //
        0.5 as f32, 1.0, 0.5, //
        1.0, 4.0, 5.0, //
    ];
    let left = dev.htod_copy(left_data.clone()).unwrap();
    let right = dev.htod_copy(right_data.clone()).unwrap();
    let mut gpu_out = dev.alloc_zeros::<f32>(1).unwrap();

    let kernel = dev.get_func("dot_kernel_module", "dot_kernel").unwrap();
    let cfg = LaunchConfig::for_num_elems(n);
    unsafe { kernel.launch(cfg, (&left, &right, &mut gpu_out, n)) }.unwrap();

    let out_host: Vec<f32> = dev.dtoh_sync_copy(&gpu_out).unwrap();

    let cpu_output = {
        let device = Device::cpu();
        let left = device.tensor(1, n as usize, left_data).unwrap();
        let right = device.tensor(1, n as usize, right_data).unwrap();
        let output = device.tensor(1, 1 as usize, vec![0.0]).unwrap();
        device.dot(&left, &right, &output).unwrap();
        output.get_values().unwrap()
    };

    let precision = 10e-7;
    let expected = cpu_output
        .into_iter()
        .map(|x| ((x / precision).round()) * precision)
        .collect::<Vec<_>>();

    let actual = out_host
        .into_iter()
        .map(|x| ((x / precision).round()) * precision)
        .collect::<Vec<_>>();

    assert_eq!(expected, actual,);
}

#[test]
fn cuda_dot_kernel_big_vectors() {
    use crate::CudaDev;
    use crate::{Device, DeviceInterface};
    use cudarc::driver::{LaunchAsync, LaunchConfig};
    use rand::Rng;

    let cuda_device = CudaDev::try_default().unwrap();
    let dev = cuda_device.dev;

    let n = 1000000; // Larger than block_dim, which is 1024 in cudarc
    let mut rng = rand::thread_rng();

    let left_data: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0..1.0)).collect();
    let right_data: Vec<f32> = (0..n).map(|_| rng.gen_range(0.0..1.0)).collect();

    let left = dev.htod_copy(left_data.clone()).unwrap();
    let right = dev.htod_copy(right_data.clone()).unwrap();
    let mut gpu_out = dev.alloc_zeros::<f32>(1).unwrap();

    let kernel = dev.get_func("dot_kernel_module", "dot_kernel").unwrap();
    let cfg = LaunchConfig::for_num_elems(n);
    unsafe { kernel.launch(cfg, (&left, &right, &mut gpu_out, n)) }.unwrap();

    let out_host: Vec<f32> = dev.dtoh_sync_copy(&gpu_out).unwrap();

    let cpu_output = {
        let device = Device::cpu();
        let left = device.tensor(1, n as usize, left_data).unwrap();
        let right = device.tensor(1, n as usize, right_data).unwrap();
        let output = device.tensor(1, 1 as usize, vec![0.0]).unwrap();
        device.dot(&left, &right, &output).unwrap();
        output.get_values().unwrap()
    };

    let precision = 1e-3;
    let expected = cpu_output
        .into_iter()
        .map(|x| ((x / precision).round()) * precision)
        .collect::<Vec<_>>()[0];

    let actual = out_host
        .into_iter()
        .map(|x| ((x / precision).round()) * precision)
        .collect::<Vec<_>>()[0];

    assert_ge!(2.0, expected - actual,);
}

#[test]
fn cuda_cross_entropy_loss() {
    use crate::CudaDev;
    use crate::{Device, DeviceInterface};
    use cudarc::driver::{LaunchAsync, LaunchConfig};

    let cuda_device = CudaDev::try_default().unwrap();
    let dev = cuda_device.dev;

    let n = 4;
    let left_data: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0];
    let right_data: Vec<f32> = vec![0.1, 0.2, 0.5, 0.2];

    let left = dev.htod_copy(left_data.clone()).unwrap();
    let right = dev.htod_copy(right_data.clone()).unwrap();
    let mut gpu_out = dev.alloc_zeros::<f32>(1).unwrap();

    let kernel = dev
        .get_func(
            "cross_entropy_loss_kernel_module",
            "cross_entropy_loss_kernel",
        )
        .unwrap();
    let cfg = LaunchConfig::for_num_elems(n);
    unsafe { kernel.launch(cfg, (&left, &right, &mut gpu_out, n, EPSILON)) }.unwrap();

    let out_host: Vec<f32> = dev.dtoh_sync_copy(&gpu_out).unwrap();

    let cpu_output = {
        let device = Device::cpu();
        let left = device.tensor(1, n as usize, left_data).unwrap();
        let right = device.tensor(1, n as usize, right_data).unwrap();
        let output = device.tensor(1, 1 as usize, vec![0.0]).unwrap();
        device.cross_entropy_loss(&left, &right, &output).unwrap();
        output.get_values().unwrap()
    };

    let precision = 10e-7;
    let expected = cpu_output
        .into_iter()
        .map(|x| ((x / precision).round()) * precision)
        .collect::<Vec<_>>();

    let actual = out_host
        .into_iter()
        .map(|x| ((x / precision).round()) * precision)
        .collect::<Vec<_>>();

    assert_eq!(expected, actual,);
}
