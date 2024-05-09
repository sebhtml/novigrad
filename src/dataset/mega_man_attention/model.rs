use std::ops::Deref;

use crate::{
    Device, Embedding, Error, ErrorEnum, Linear, MaskedScaledDotProductAttention, MatMul,
    OperatorTrait, Reshape, Softmax, Tensor, TensorF32,
};

pub struct Model {
    vocab_size: usize,
    sequence_length: usize,
    embedding: Embedding,
    q: Linear,
    k: Linear,
    v: Linear,
    attention: MaskedScaledDotProductAttention,
    linear: Linear,
    softmax: Softmax,
}

impl Model {
    pub fn new(device: &Device) -> Self {
        let _batch_size = 1;
        let sequence_length = 6;
        let vocab_size = 20;
        //let vocab_size = 34816; // 32768 + 2048
        let embedding_dim = 4;
        let _num_heads = 1;

        let q = Linear::new(device, embedding_dim, embedding_dim, sequence_length);
        let k = Linear::new(device, embedding_dim, embedding_dim, sequence_length);
        let v = Linear::new(device, embedding_dim, embedding_dim, sequence_length);

        let attention =
            MaskedScaledDotProductAttention::try_new(device, sequence_length, embedding_dim)
                .unwrap();

        let linear = Linear::new(device, vocab_size, embedding_dim, sequence_length);

        Self {
            vocab_size,
            sequence_length,
            embedding: Embedding::new(device, vocab_size, embedding_dim),
            q,
            k,
            v,
            attention,
            linear,
            softmax: Softmax::new(device, true),
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn sequence_length(&self) -> usize {
        self.sequence_length
    }
}

impl OperatorTrait for Model {
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, Error> {
        let debug = false;
        if debug {
            println!("----");
        }
        if debug {
            println!("input {}", inputs[0].tensor().deref().borrow());
        }
        let embeddings = self.embedding.forward(inputs)?;
        if debug {
            embeddings.realize()?;
            println!("embedding {}", &embeddings.tensor().deref().borrow());
        }
        let q = self.q.forward(&[embeddings.clone()])?;
        if debug {
            q.realize()?;
            println!("q {}", &q.tensor().deref().borrow());
        }
        let k = self.k.forward(&[embeddings.clone()])?;
        if debug {
            k.realize()?;
            println!("k {}", &k.tensor().deref().borrow());
        }
        let v = self.v.forward(&[embeddings.clone()])?;
        if debug {
            v.realize()?;
            println!("v {}", &v.tensor().deref().borrow());
        }
        let attended = self.attention.forward(&[q, k, v])?;
        if debug {
            attended.realize()?;
            println!("attended {}", &attended.tensor().deref().borrow());
        }
        let linearized = self.linear.forward(&[attended])?;
        if debug {
            linearized.realize()?;
            println!("linearized {}", &linearized.tensor().deref().borrow());
        }
        let probabilities = self.softmax.forward(&[linearized])?;
        if debug {
            probabilities.realize()?;
            println!("probabilities {}", &probabilities.tensor().deref().borrow());
        }
        Ok(probabilities)
    }

    fn name(&self) -> &str {
        "MegaManModel"
    }

    fn backward(&self, _inputs: &[Tensor], _output: &Tensor) -> Result<(), Error> {
        Err(Error::new(
            file!(),
            line!(),
            column!(),
            ErrorEnum::UnsupportedOperation,
        ))
    }

    fn forward_realize(&self, _inputs: &[Tensor], output: &Tensor) -> Result<(), Error> {
        Err(Error::new(
            file!(),
            line!(),
            column!(),
            ErrorEnum::UnsupportedOperation,
        ))
    }
}
