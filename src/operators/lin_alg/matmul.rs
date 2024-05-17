use std::{ops::Deref, rc::Rc};

use crate::{devices::Device, BinaryOperator, Error, ErrorEnum, Gemm, Tensor, TensorF32, Zero};

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
        output.push_instruction(
            Rc::new(Zero::default()),
            &[],
            &[&outputs[0].tensor().deref().borrow()],
            false,
        );
        output.push_instruction(
            Rc::new(Zero::default()),
            &[],
            &[&outputs[0].gradient().deref().borrow()],
            false,
        );
        output.push_instruction(
            Rc::new(Gemm::new(&self.device, false, transb, false)),
            &[
                &inputs[0].tensor().deref().borrow(),
                &inputs[1].tensor().deref().borrow(),
            ],
            &[&outputs[0].tensor().deref().borrow()],
            false,
        );

        if input_1.gradient().deref().borrow().requires_grad() {
            output.push_backward_instruction(
                Rc::new(Gemm::new(&self.device, true, false, transb)),
                &[
                    &input_0.tensor().deref().borrow(),
                    &output.gradient().deref().borrow(),
                ],
                &[&input_1.gradient().deref().borrow()],
                true,
            );
        }

        if input_0.gradient().deref().borrow().requires_grad() {
            output.push_backward_instruction(
                Rc::new(Gemm::new(&self.device, true, transb, true)),
                &[
                    &input_1.tensor().deref().borrow(),
                    &output.gradient().deref().borrow(),
                ],
                &[&input_0.gradient().deref().borrow()],
                true,
            );
        }

        Ok(output)
    }
}
