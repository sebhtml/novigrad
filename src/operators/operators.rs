use crate::{
    CrossEntropyLoss, Device, Embedding, Identity, Linear, MatMul, Reshape, ResidualSumOfSquares,
    Sigmoid, Softmax,
};

pub struct Operators {
    device: Device,
}

impl Operators {
    pub fn new(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }
}

impl Operators {
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn embedding(&self, num_embeddings: usize, embedding_dim: usize) -> Embedding {
        Embedding::new(num_embeddings, embedding_dim, &self.device)
    }

    pub fn matmul(&self) -> MatMul {
        MatMul::new(&self.device)
    }

    pub fn reshape(
        &self,
        input_rows: usize,
        input_cols: usize,
        output_rows: usize,
        output_cols: usize,
    ) -> Reshape {
        Reshape::new(
            &self.device,
            input_rows,
            input_cols,
            output_rows,
            output_cols,
        )
    }

    pub fn linear(&self, weights_rows: usize, weights_cols: usize, bias_rows: usize) -> Linear {
        Linear::new(weights_rows, weights_cols, bias_rows, &self.device)
    }

    pub fn sigmoid(&self) -> Sigmoid {
        Sigmoid::new(&self.device)
    }

    pub fn softmax(&self, using_cross_entropy_loss: bool) -> Softmax {
        Softmax::new(using_cross_entropy_loss, &self.device)
    }

    pub fn residual_sum_of_squares(&self) -> ResidualSumOfSquares {
        ResidualSumOfSquares::new(&self.device)
    }

    pub fn cross_entropy_loss(&self) -> CrossEntropyLoss {
        CrossEntropyLoss::new(&self.device)
    }

    pub fn identity(&self) -> Identity {
        Identity::new(&self.device)
    }
}
