use std::{ops::Deref, rc::Rc};

use crate::{Device, Error, Identity, Mul, OperatorTrait, Tensor};

/// Linear is not a ONNX operator. https://onnx.ai/onnx/operators/index.html ???
/// Attention Is All You Need -> https://arxiv.org/abs/1706.03762
#[derive(Clone)]
pub struct Mask {
    mask: Tensor,
    mul: Mul,
}

impl Mask {
    pub fn try_new(device: &Device, mask_rows: usize, mask_cols: usize) -> Result<Self, Error> {
        let len = mask_rows * mask_cols;
        let mask = vec![1.0; len];

        let mask = device.tensor(
            Rc::new(Identity::new(device)),
            &vec![],
            mask_rows,
            mask_cols,
            mask,
            true,
            true,
        );
        let mut values = mask.tensor().deref().borrow().get_values()?;
        for row in 0..mask_rows {
            for col in 0..mask_cols {
                if row > col {
                    let index = mask.tensor().deref().borrow().index(row, col);
                    values[index] = f32::NEG_INFINITY;
                }
            }
        }
        mask.tensor().deref().borrow_mut().set_values(values);

        /*
        {
            let mask: &TensorF32 = &mask.tensor().deref().borrow();
            println!("mask {}", mask);
        }
        */
        let mul = Mul::new(device);
        let mask = Self { mask, mul };
        Ok(mask)
    }
}

impl OperatorTrait for Mask {
    fn name(&self) -> &str {
        "Mask"
    }

    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, Error> {
        let inputs = &[inputs[0].clone(), self.mask.clone()];
        self.mul.forward(inputs)
    }

    fn forward_realize(&self, inputs: &[Tensor], output: &Tensor) -> Result<(), Error> {
        self.mul.forward_realize(inputs, output)
    }

    fn backward(&self, inputs: &[Tensor], output: &Tensor) -> Result<(), Error> {
        self.mul.backward(inputs, output)
    }
}
