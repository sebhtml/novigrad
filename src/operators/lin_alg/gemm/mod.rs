use std::ops::Deref;

use crate::{devices::Device, error, DeviceInterface, Error, ErrorEnum, Tensor};

#[cfg(test)]
mod tests;
pub struct Gemm {}

impl Gemm {
    pub fn new(_device: &Device) -> Self {
        Self {}
    }

    pub fn execute(
        trans_a: bool,
        trans_b: bool,
        trans_result: bool,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
    ) -> Result<(), Error> {
        debug_assert_eq!(inputs.len(), 2);
        debug_assert_eq!(outputs.len(), 1);
        let input = inputs[0];
        let weights = inputs[1];
        let biases = outputs[0];
        let a = input;
        let b = weights;
        let c = biases;
        let transa = trans_a;
        let transb = trans_b;
        let transpose_result = trans_result;
        let alpha = 1.0;
        let beta = 1.0;
        Gemm::gemm(transa, transb, alpha, a, b, beta, c, transpose_result)
    }

    pub fn gemm(
        transa: bool,
        transb: bool,
        alpha: f32,
        a: &Tensor,
        b: &Tensor,
        beta: f32,
        c: &Tensor,
        transpose_result: bool,
    ) -> Result<(), Error> {
        let op_result = Self::_gemm(transa, transb, alpha, a, b, beta, c, transpose_result);
        match op_result {
            Ok(value) => Ok(value),
            Err(error) => {
                println!("Incompatible sizes in GEMM");
                println!(
                    "transa: {}, transb: {}, transpose_result: {}",
                    transa, transb, transpose_result
                );
                println!(
                    "A size: {:?}  B size:  {:?}  C size:  {:?}",
                    a.size().deref().borrow(),
                    b.size().deref().borrow(),
                    c.size().deref().borrow(),
                );
                debug_assert!(false);
                Err(error)
            }
        }
    }

    fn _gemm(
        transa: bool,
        transb: bool,
        alpha: f32,
        a: &Tensor,
        b: &Tensor,
        beta: f32,
        c: &Tensor,
        transpose_result: bool,
    ) -> Result<(), Error> {
        let device = a.device();
        if !transa && !transb && !transpose_result {
            if a.cols() != b.rows() {
                return Err(error!(ErrorEnum::IncompatibleTensorShapes));
            }
            if a.rows() != c.rows() {
                return Err(error!(ErrorEnum::IncompatibleTensorShapes));
            }
            if b.cols() != c.cols() {
                return Err(error!(ErrorEnum::IncompatibleTensorShapes));
            }
            let (m, n, k) = (a.rows(), b.cols(), a.cols());
            device.gemm(
                false, false, n as i32, m as i32, k as i32, alpha, b, n as i32, a, k as i32, beta,
                c, n as i32,
            )
        } else if transa && !transb && !transpose_result {
            if a.rows() != b.rows() {
                return Err(error!(ErrorEnum::IncompatibleTensorShapes));
            }
            if a.cols() != c.rows() {
                return Err(error!(ErrorEnum::IncompatibleTensorShapes));
            }
            if b.cols() != c.cols() {
                return Err(error!(ErrorEnum::IncompatibleTensorShapes));
            }

            let (m, n, k) = (a.cols(), b.cols(), a.rows());

            device.gemm(
                false,
                true,
                n as i32,
                m as i32,
                k as i32,
                alpha,
                b,
                n as i32,
                a,
                a.cols() as i32,
                beta,
                c,
                n as i32,
            )
        } else if !transa && transb && !transpose_result {
            if a.cols() != b.cols() {
                return Err(error!(ErrorEnum::IncompatibleTensorShapes));
            }
            if a.rows() != c.rows() {
                return Err(error!(ErrorEnum::IncompatibleTensorShapes));
            }
            if b.rows() != c.cols() {
                return Err(error!(ErrorEnum::IncompatibleTensorShapes));
            }
            let (m, n, k) = (a.rows(), b.rows(), a.cols());

            device.gemm(
                true,
                false,
                n as i32,
                m as i32,
                k as i32,
                alpha,
                b,
                b.cols() as i32,
                a,
                k as i32,
                beta,
                c,
                n as i32,
            )
        } else if transa && transb && !transpose_result {
            if a.rows() != b.cols() {
                return Err(error!(ErrorEnum::IncompatibleTensorShapes));
            }
            if a.cols() != c.rows() {
                return Err(error!(ErrorEnum::IncompatibleTensorShapes));
            }
            if b.rows() != c.cols() {
                return Err(error!(ErrorEnum::IncompatibleTensorShapes));
            }
            let (m, n, k) = (a.cols(), b.rows(), a.rows());

            device.gemm(
                true,
                true,
                n as i32,
                m as i32,
                k as i32,
                alpha,
                b,
                b.cols() as i32,
                a,
                a.cols() as i32,
                beta,
                c,
                n as i32,
            )
        } else if transa && transb && transpose_result {
            if a.rows() != b.cols() {
                return Err(error!(ErrorEnum::IncompatibleTensorShapes));
            }
            if a.cols() != c.cols() {
                return Err(error!(ErrorEnum::IncompatibleTensorShapes));
            }
            if b.rows() != c.rows() {
                return Err(error!(ErrorEnum::IncompatibleTensorShapes));
            }
            let (m, n, k) = (a.cols(), b.rows(), a.rows());

            device.gemm(
                false,
                false,
                m as i32,
                n as i32,
                k as i32,
                alpha,
                a,
                a.cols() as i32,
                b,
                b.cols() as i32,
                beta,
                c,
                m as i32,
            )
        } else if transa && !transb && transpose_result {
            if a.rows() != b.rows() {
                return Err(error!(ErrorEnum::IncompatibleTensorShapes));
            }
            if a.cols() != c.cols() {
                return Err(error!(ErrorEnum::IncompatibleTensorShapes));
            }
            if b.cols() != c.rows() {
                return Err(error!(ErrorEnum::IncompatibleTensorShapes));
            }
            let (m, n, k) = (a.cols(), b.cols(), a.rows());

            device.gemm(
                false,
                true,
                m as i32,
                n as i32,
                k as i32,
                alpha,
                a,
                a.cols() as i32,
                b,
                b.cols() as i32,
                beta,
                c,
                m as i32,
            )
        } else {
            Err(error!(ErrorEnum::UnsupportedOperation))
        }
    }
}
