use std::ops::Deref;

use crate::{
    CausalSelfAttention, Device, Embedding, Error, Linear, Model, OperatorTrait, Softmax, Tensor,
};

pub struct MegaManAttentionModel {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    vocab_size: usize,
    sequence_length: usize,
    embedding: Embedding,
    q: Linear,
    k: Linear,
    v: Linear,
    attention: CausalSelfAttention,
    linear: Linear,
    softmax: Softmax,
}

impl MegaManAttentionModel {
    pub fn new(device: &Device) -> Self {
        let _batch_size = 1;
        let sequence_length = 6;
        let vocab_size = 20;
        let n_embd = 4;
        let _num_heads = 1;
        let _n_layer = 1;
        let _dropout = 0.1;
        let _block_size = 2048;

        let q = Linear::new(device, n_embd, n_embd, sequence_length);
        let k = Linear::new(device, n_embd, n_embd, sequence_length);
        let v = Linear::new(device, n_embd, n_embd, sequence_length);

        let attention = CausalSelfAttention::try_new(device, sequence_length, n_embd).unwrap();

        let linear = Linear::new(device, vocab_size, n_embd, sequence_length);

        Self {
            input_shape: vec![sequence_length, vocab_size],
            output_shape: vec![sequence_length, vocab_size],
            vocab_size,
            sequence_length,
            embedding: Embedding::new(device, vocab_size, n_embd),
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

impl Model for MegaManAttentionModel {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, Error> {
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
        let q = self.q.forward(&[&embeddings])?;
        if debug {
            q.realize()?;
            println!("q {}", &q.tensor().deref().borrow());
        }
        let k = self.k.forward(&[&embeddings])?;
        if debug {
            k.realize()?;
            println!("k {}", &k.tensor().deref().borrow());
        }
        let v = self.v.forward(&[&embeddings])?;
        if debug {
            v.realize()?;
            println!("v {}", &v.tensor().deref().borrow());
        }
        let attended = self.attention.forward(&[&q, &k, &v])?;
        if debug {
            attended.realize()?;
            println!("attended {}", &attended.tensor().deref().borrow());
        }
        let linearized = self.linear.forward(&[&attended])?;
        if debug {
            linearized.realize()?;
            println!("linearized {}", &linearized.tensor().deref().borrow());
        }
        let probabilities = self.softmax.forward(&[&linearized])?;
        if debug {
            probabilities.realize()?;
            println!("probabilities {}", &probabilities.tensor().deref().borrow());
        }
        Ok(probabilities)
    }

    fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }
}
