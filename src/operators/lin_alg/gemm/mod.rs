use crate::{
    devices::Device,
    error,
    stream::DeviceStream,
    tensor::{Error, ErrorEnum, Tensor},
    DeviceTrait, ExecutableOperator, OperatorAttributes,
};

#[cfg(test)]
mod tests;
pub struct Gemm {}

impl Gemm {
    pub fn new(_device: &Device) -> Self {
        Self {}
    }
}

impl ExecutableOperator for Gemm {
    /// C := alpha*op( A )*op( B ) + beta*C,
    fn execute(
        attributes: &OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        debug_assert_eq!(inputs.len(), 3);
        debug_assert_eq!(outputs.len(), 1);
        let input = inputs[0]; // A
        let weights = inputs[1]; // B
        let _ = inputs[2]; // C
        let biases = outputs[0]; // C
        let a = input;
        let b = weights;
        let c = biases;
        let (transa, transb, transpose_result) = match attributes {
            OperatorAttributes::ThreeBools(transa, transb, transpose_result) => {
                (*transa, *transb, *transpose_result)
            }
            _ => {
                return Err(error!(ErrorEnum::UnsupportedOperation));
            }
        };
        let alpha = 1.0;
        let beta = 1.0;
        Gemm::gemm(
            transa,
            transb,
            alpha,
            a,
            b,
            beta,
            c,
            transpose_result,
            device_stream,
        )
    }
}

impl Gemm {
    pub fn gemm(
        transa: bool,
        transb: bool,
        alpha: f32,
        a: &Tensor,
        b: &Tensor,
        beta: f32,
        c: &Tensor,
        transpose_result: bool,
        device_stream: &DeviceStream,
    ) -> Result<(), Error> {
        let op_result = Self::_gemm(
            transa,
            transb,
            alpha,
            a,
            b,
            beta,
            c,
            transpose_result,
            device_stream,
        );
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
                    *a.size(),
                    *b.size(),
                    *c.size(),
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
        device_stream: &DeviceStream,
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
                false,
                false,
                n as i32,
                m as i32,
                k as i32,
                alpha,
                b.as_ptr(),
                n as i32,
                a.as_ptr(),
                k as i32,
                beta,
                c.as_mut_ptr(),
                n as i32,
                device_stream,
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
                b.as_ptr(),
                n as i32,
                a.as_ptr(),
                a.cols() as i32,
                beta,
                c.as_mut_ptr(),
                n as i32,
                device_stream,
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
                b.as_ptr(),
                b.cols() as i32,
                a.as_ptr(),
                k as i32,
                beta,
                c.as_mut_ptr(),
                n as i32,
                device_stream,
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
                b.as_ptr(),
                b.cols() as i32,
                a.as_ptr(),
                a.cols() as i32,
                beta,
                c.as_mut_ptr(),
                n as i32,
                device_stream,
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
                a.as_ptr(),
                a.cols() as i32,
                b.as_ptr(),
                b.cols() as i32,
                beta,
                c.as_mut_ptr(),
                m as i32,
                device_stream,
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
                a.as_ptr(),
                a.cols() as i32,
                b.as_ptr(),
                b.cols() as i32,
                beta,
                c.as_mut_ptr(),
                m as i32,
                device_stream,
            )
        } else {
            Err(error!(ErrorEnum::UnsupportedOperation))
        }
    }
}
