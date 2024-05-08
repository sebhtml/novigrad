use std::ops::Deref;

use crate::{
    Device, Embedding, Error, Linear, MaskedScaledDotProductAttention, MatMul, OperatorTrait,
    Reshape, Softmax, Tensor,
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
        let sequence_length = 32;
        let vocab_size = 256;
        //let vocab_size = 34816; // 32768 + 2048
        let embedding_dim = 384;
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
        let embeddings = self.embedding.forward(inputs)?;
        //embeddings.realize()?;
        //println!(
        //    "embedding {:?}",
        //    embeddings.tensor().deref().borrow().size()
        //);
        let q = self.q.forward(&[embeddings.clone()])?;
        //q.realize()?;
        //println!("q {:?}", q.tensor().deref().borrow().size());
        let k = self.k.forward(&[embeddings.clone()])?;
        //k.realize()?;
        //println!("k {:?}", k.tensor().deref().borrow().size());
        let v = self.v.forward(&[embeddings.clone()])?;
        //v.realize()?;
        //println!("v {:?}", v.tensor().deref().borrow().size());
        let attended = self.attention.forward(&[q, k, v])?;
        //attended.realize()?;
        //println!("attended {:?}", attended.tensor().deref().borrow().size());
        let linearized = self.linear.forward(&[attended])?;
        //linearized.realize()?;
        //println!("linearized {:?}", linearized.tensor().deref().borrow().size());
        let probabilities = self.softmax.forward(&[linearized])?;
        //probabilities.realize()?;
        //println!("probabilities {:?}", probabilities.tensor().deref().borrow().size());
        Ok(probabilities)
    }

    fn name(&self) -> &str {
        "MegaManModel"
    }

    fn backward(&self, _inputs: &[Tensor], _output: &Tensor) -> Result<(), Error> {
        Err(Error::UnsupportedOperation)
    }

    fn forward_realize(&self, _inputs: &[Tensor], output: &Tensor) -> Result<(), Error> {
        Err(Error::UnsupportedOperation)
    }
}
