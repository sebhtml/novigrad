use std::{ops::Deref, rc::Rc};

use crate::{Add, Device, Error, Identity, Operator, Tensor};

/// Linear is not a ONNX operator. https://onnx.ai/onnx/operators/index.html ???
/// Attention Is All You Need -> https://arxiv.org/abs/1706.03762
#[derive(Clone)]
pub struct Mask {
    mask: Tensor,
    add: Add,
}

impl Mask {
    pub fn try_new(device: &Device, mask_rows: usize, mask_cols: usize) -> Result<Self, Error> {
        let len = mask_rows * mask_cols;
        let mask = vec![0.0; len];

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
                if row < col {
                    let index = mask.tensor().deref().borrow().index(row, col);
                    values[index] = f32::NEG_INFINITY;
                }
            }
        }
        mask.tensor().deref().borrow_mut().set_values(values);

        {
            println!("mask {}", &mask.tensor().deref().borrow());
        }

        let add = Add::new(device);
        let mask = Self { mask, add };
        Ok(mask)
    }

    pub fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, Error> {
        let inputs = &[&inputs[0], &self.mask];
        self.add.forward(inputs)
    }
}
