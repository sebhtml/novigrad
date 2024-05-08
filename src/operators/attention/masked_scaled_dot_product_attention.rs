use std::{ops::Deref, rc::Rc};

use crate::{Device, Error, Gemm, Identity, Mask, MatMul, OperatorTrait, Scale, Softmax, Tensor};

/// MaskedScaledDotProductAttention is not a ONNX operator.
/// https://onnx.ai/onnx/operators/index.html ???
/// Attention Is All You Need
/// https://arxiv.org/abs/1706.03762
#[derive(Clone)]
pub struct MaskedScaledDotProductAttention {
    gemm: Gemm, // The MatMul in "Attention Is All You Need" in Fig. 2 on page 4
    gemm_biases: Tensor,
    scale: Scale,
    mask: Mask,
    softmax: Softmax,
    matmul: MatMul,
}

impl MaskedScaledDotProductAttention {
    pub fn try_new(device: &Device, rows: usize, cols: usize) -> Result<Self, Error> {
        let gemm = Gemm::new(device);

        let len = rows * rows;
        let gemm_biases = device.tensor(
            Rc::new(Identity::new(device)),
            &vec![],
            rows,
            rows,
            vec![0.0; len],
            true,
            true,
        );

        let alpha = 1.0 / f32::sqrt(cols as f32);
        let scale = Scale::new(device, alpha);
        let mask_rows = rows;
        let mask_cols = rows;
        let mask = Mask::try_new(device, mask_rows, mask_cols)?;
        let next_op_is_cross_entropy_loss = false;
        let softmax = Softmax::new(device, next_op_is_cross_entropy_loss);
        let transb = false;
        let matmul = MatMul::new(device, transb);

        let attention = Self {
            gemm,
            gemm_biases,
            scale,
            mask,
            softmax,
            matmul,
        };
        Ok(attention)
    }
}

impl OperatorTrait for MaskedScaledDotProductAttention {
    fn name(&self) -> &str {
        "MaskedScaledDotProductAttention"
    }

    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, Error> {
        debug_assert_eq!(inputs.len(), 3);
        let q = &inputs[0];
        let k = &inputs[1];
        let v = &inputs[2];
        /*
        println!(
            "q {:?} k {:?} v {:?}",
            q.tensor().deref().borrow().size(),
            k.tensor().deref().borrow().size(),
            v.tensor().deref().borrow().size(),
        );
         */
        //println!("q {}", q.tensor().deref().borrow());
        let weights = self
            .gemm
            .forward(&[q.clone(), k.clone(), self.gemm_biases.clone()])?;
        //weights.realize()?;
        //println!("weights {}", weights.tensor().deref().borrow());
        let scaled_weights = self.scale.forward(&[weights])?;
        //scaled_weights.realize()?;
        //println!("scaled_weights {}", scaled_weights.tensor().deref().borrow());
        let masked_weights = self.mask.forward(&[scaled_weights])?;
        //masked_weights.realize()?;
        //println!("masked_weights {}", masked_weights.tensor().deref().borrow());
        //println!(
        //    "masked_weights {:?}",
        //    masked_weights.tensor().deref().borrow().size()
        //);
        let softmaxed_weights = self.softmax.forward(&[masked_weights])?;
        //softmaxed_weights.realize()?;
        //println!("softmaxed_weights {}", softmaxed_weights.tensor().deref().borrow());
        //println!(
        //"softmaxed_weights {:?}",
        //softmaxed_weights.tensor().deref().borrow().size()
        //);
        let attentions = self.matmul.forward(&[softmaxed_weights, v.clone()])?;
        //println!(
        //"attentions {:?}",
        //attentions.tensor().deref().borrow().size()
        //);
        Ok(attentions)
    }

    fn forward_realize(&self, _inputs: &[Tensor], _output: &Tensor) -> Result<(), Error> {
        Err(Error::UnsupportedOperation)
    }

    fn backward(&self, _inputs: &[Tensor], _output: &Tensor) -> Result<(), Error> {
        Err(Error::UnsupportedOperation)
    }
}
