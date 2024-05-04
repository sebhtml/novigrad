use std::ops::Deref;

use crate::{devices::Device, Error, OperatorTrait, Tensor, TensorF32};

pub struct MatMul {}

impl MatMul {
    pub fn new() -> Self {
        MatMul {}
    }

    fn forward(device: &Device, inputs: &[Tensor], output: &mut Tensor) -> Result<(), Error> {
        debug_assert_eq!(inputs.len(), 2);
        let input_0: &TensorF32 = &inputs[0].tensor().deref().borrow();
        let input_1: &TensorF32 = &inputs[1].tensor().deref().borrow();

        {
            let output: &mut TensorF32 = &mut output.tensor().deref().borrow_mut();
            let a = input_0;
            let b = input_1;
            let c = output;
            let op_result = TensorF32::gemm(device, false, true, 1.0, a, b, 1.0, c, false);
            match op_result {
                Ok(_) => (),
                Err(_) => {
                    let mut b_t =
                        device.tensor_f32(b.cols(), b.rows(), vec![0.0; b.cols() * b.rows()]);
                    b.transpose(&mut b_t)?;
                    println!("Incompatible shapes in matrix multiplication");
                    println!("Between A {:?} and B^T {:?}", a.shape(), b_t.shape(),);
                    debug_assert!(false);
                }
            }
        }

        Ok(())
    }

    fn backward(device: &Device, inputs: &[Tensor], output: &Tensor) -> Result<(), Error> {
        debug_assert_eq!(inputs.len(), 2);
        let output_gradient: &TensorF32 = &output.gradient().deref().borrow();
        {
            let input_1_gradient: &mut TensorF32 = &mut inputs[1].gradient().deref().borrow_mut();
            let input_0: &TensorF32 = &inputs[0].tensor().deref().borrow();
            let a: &TensorF32 = input_0;
            let b: &TensorF32 = output_gradient;
            let c: &mut TensorF32 = input_1_gradient;
            TensorF32::gemm(device, true, false, 1.0, a, b, 1.0, c, true)?;
        }

        {
            let input_0_gradient: &mut TensorF32 = &mut inputs[0].gradient().deref().borrow_mut();
            let input_1: &TensorF32 = &inputs[1].tensor().deref().borrow();
            let a: &TensorF32 = input_1;
            let b: &TensorF32 = output_gradient;
            let c: &mut TensorF32 = input_0_gradient;
            TensorF32::gemm(device, true, true, 1.0, a, b, 1.0, c, true)?;
        }

        Ok(())
    }
}

impl OperatorTrait for MatMul {
    fn forward(&self, device: &Device, inputs: &[Tensor]) -> Result<Tensor, Error> {
        let input_0: &TensorF32 = &inputs[0].tensor().deref().borrow();
        let input_1: &TensorF32 = &inputs[1].tensor().deref().borrow();
        let rows = input_0.rows();
        let cols = input_1.rows();
        let len = rows * cols;
        let mut output = device.tensor(inputs, rows, cols, vec![0.0; len], false);
        MatMul::forward(device, inputs, &mut output)?;
        Ok(output)
    }

    fn name(&self) -> &str {
        "MatMul"
    }

    fn backward(&self, device: &Device, inputs: &[Tensor], output: &Tensor) -> Result<(), Error> {
        MatMul::backward(device, inputs, output)
    }
}
