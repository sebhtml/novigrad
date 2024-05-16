use std::{ops::Deref, rc::Rc};

use crate::{devices::Device, BinaryOperator, Error, ErrorEnum, Operator, Tensor, TensorF32, Zero};

/// https://onnx.ai/onnx/operators/onnx__MatMul.html
#[derive(Clone)]
pub struct MatMul {
    device: Device,
    transb: bool,
}

impl MatMul {
    pub fn new(device: &Device, transb: bool) -> Self {
        MatMul {
            device: device.clone(),
            transb,
        }
    }
}

impl BinaryOperator for MatMul {
    fn forward(&self, input_0: &Tensor, input_1: &Tensor) -> Result<Tensor, Error> {
        let input_0_tensor: &TensorF32 = &input_0.tensor().deref().borrow();
        let input_1_tensor: &TensorF32 = &input_1.tensor().deref().borrow();
        /*println!("a size {:?}, b size {:?} transb {}",
            input_0_tensor.size().deref().borrow(),
                    input_1_tensor.size().deref().borrow(),
                self.transb);
        */
        let compatible = match self.transb {
            false => input_0_tensor.cols() == input_1_tensor.rows(),
            true => input_0_tensor.cols() == input_1_tensor.cols(),
        };
        if !compatible {
            println!("Incompatible shapes in matrix multiplication");
            println!("transa: {}, transb: {}", false, self.transb);
            println!(
                "Between A {:?} and B {:?}",
                input_0_tensor.size().deref().borrow(),
                input_1_tensor.size().deref().borrow(),
            );
            debug_assert!(false);
            return Err(Error::new(
                file!(),
                line!(),
                column!(),
                ErrorEnum::IncompatibleTensorShapes,
            ));
        }

        let rows = input_0_tensor.rows();
        let transb = self.transb;
        let cols = if transb {
            input_1_tensor.rows()
        } else {
            input_1_tensor.cols()
        };
        let len = rows * cols;
        let output =
            self.device
                .tensor(rows, cols, vec![0.0; len], &[input_0, input_1], true, false);

        let inputs = [input_0, input_1];
        let outputs = [&output];
        output.push_forward_instruction_f32(
            Rc::new(Zero::default()),
            &[],
            &[&outputs[0].tensor().deref().borrow()],
        ); //
        output.push_forward_instruction_f32(
            Rc::new(Zero::default()),
            &[],
            &[&outputs[0].gradient().deref().borrow()],
        ); //
        output.push_forward_instruction_f32(
            Rc::new(self.clone()),
            &[
                &inputs[0].tensor().deref().borrow(),
                &inputs[1].tensor().deref().borrow(),
            ],
            &[&outputs[0].tensor().deref().borrow()],
        );

        let inputs = [input_0, input_1, &output];
        let outputs = [input_0, input_1];
        output.push_backward_instruction_f32(
            Rc::new(MatMulBackward::new(self.transb)),
            &[
                &inputs[0].tensor().deref().borrow(),
                &inputs[1].tensor().deref().borrow(),
                &inputs[2].gradient().deref().borrow(),
            ],
            &[
                &outputs[0].gradient().deref().borrow(),
                &outputs[1].gradient().deref().borrow(),
            ],
        );
        Ok(output)
    }
}

impl Operator for MatMul {
    fn name(&self) -> &str {
        "MatMul"
    }

    fn forward(&self, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        self.forward_f32(
            &[
                &inputs[0].tensor().deref().borrow(),
                &inputs[1].tensor().deref().borrow(),
            ],
            &[&outputs[0].tensor().deref().borrow()],
        )
    }

    fn forward_f32(&self, inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        debug_assert_eq!(inputs.len(), 2);
        let input_0 = inputs[0];
        let input_1 = inputs[1];
        let output = outputs[0];
        let a = input_0;
        let b = input_1;
        let c = output;
        let transb = self.transb;
        TensorF32::matmul(false, transb, a, b, c, false)
    }
}

pub struct MatMulBackward {
    transb: bool,
}

impl MatMulBackward {
    pub fn new(transb: bool) -> Self {
        MatMulBackward { transb }
    }
}

impl Operator for MatMulBackward {
    fn name(&self) -> &str {
        "MatMulBackward"
    }

    fn forward(&self, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        self.forward_f32(
            &[
                &inputs[0].tensor().deref().borrow(),
                &inputs[1].tensor().deref().borrow(),
                &inputs[2].gradient().deref().borrow(),
            ],
            &[
                &outputs[0].gradient().deref().borrow(),
                &outputs[1].gradient().deref().borrow(),
            ],
        )
    }

    fn forward_f32(&self, inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        debug_assert_eq!(outputs.len(), 2);
        let input_gradient = inputs[2];

        if outputs[1].requires_grad() {
            let output_1_gradient = outputs[1];
            let output_0 = inputs[0];
            let a = output_0;
            let b = input_gradient;
            let c = output_1_gradient;
            let transb = self.transb;
            TensorF32::gemm(true, false, 1.0, a, b, 1.0, c, transb)?;
        }

        if outputs[0].requires_grad() {
            let output_0_gradient = outputs[0];
            let output_1 = inputs[1];
            let a = output_1;
            let b = input_gradient;
            let c = output_0_gradient;
            let transb = self.transb;
            if transb {
                TensorF32::gemm(true, true, 1.0, a, b, 1.0, c, true)?;
            } else {
                TensorF32::gemm(true, false, 1.0, a, b, 1.0, c, true)?;
            }
        }

        Ok(())
    }
}
