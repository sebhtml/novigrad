use crate::{Error, Forward, Operator, Operators, Tensor};

pub struct Model {
    embedding: Operator,
    linear_0: Operator,
    sigmoid_0: Operator,
    reshape: Operator,
    linear_1: Operator,
    sigmoid_1: Operator,
    linear_2: Operator,
    softmax: Operator,
}

impl Model {
    pub fn new(ops: &Operators) -> Self {
        let _batch_size = 1;
        let sequence_length = 6;
        let vocab_size = 256;
        let num_embeddings = vocab_size;
        let embedding_dim = 384;
        let _num_heads = 0;
        Self {
            embedding: ops.embedding(num_embeddings, embedding_dim),
            linear_0: ops.linear(embedding_dim, embedding_dim, sequence_length),
            sigmoid_0: ops.sigmoid(),
            reshape: ops.reshape(
                sequence_length,
                embedding_dim,
                1,
                sequence_length * embedding_dim,
            ),
            linear_1: ops.linear(embedding_dim, sequence_length * embedding_dim, 1),
            sigmoid_1: ops.sigmoid(),
            linear_2: ops.linear(vocab_size, embedding_dim, 1),
            softmax: ops.softmax(true),
        }
    }
}

impl Forward for Model {
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor, Error> {
        let state_0: Tensor = self.embedding.forward(inputs)?;
        let state_1 = self.linear_0.forward(&[state_0])?;
        let state_2 = self.sigmoid_0.forward(&[state_1])?;
        let state_3 = self.reshape.forward(&[state_2])?;
        let state_4 = self.linear_1.forward(&[state_3])?;
        let state_5 = self.sigmoid_1.forward(&[state_4])?;
        let state_6 = self.linear_2.forward(&[state_5])?;
        let state_7 = self.softmax.forward(&[state_6])?;
        Ok(state_7)
    }
}
