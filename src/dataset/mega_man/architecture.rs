use std::{borrow::Borrow, cell::RefCell, rc::Rc};

use crate::{
    Accelerator, DifferentiableModule, DifferentiableModuleConfig, EmbeddingConfig, Error, Forward,
    FullDifferentiableModuleConfig, ReshapeConfig, Session, Tape, Tensor,
};

pub struct Architecture {
    embedding: DifferentiableModule,
    reshape: DifferentiableModule,
    linear: DifferentiableModule,
    softmax: DifferentiableModule,
}

impl Default for Architecture {
    fn default() -> Self {
        let session = Session::default();
        let accelerator = session.accelerator();
        let tape = session.tape();
        let configs = architecture();
        let mut iterator = configs.iter().peekable();
        let linear = session.linear(256, 32 * 384, 1);
        let softmax = session.softmax(true);
        Self {
            embedding: FullDifferentiableModuleConfig {
                accelerator: &accelerator,
                tape: &tape,
                config: iterator.next().unwrap(),
            }
            .borrow()
            .into(),
            reshape: FullDifferentiableModuleConfig {
                accelerator: &accelerator,
                tape: &tape,
                config: iterator.next().unwrap(),
            }
            .borrow()
            .into(),
            linear,
            softmax,
        }
    }
}

impl Forward for Architecture {
    fn forward(&mut self, layer_input: &Tensor) -> Result<Tensor, Error> {
        let embedding = self.embedding.forward(layer_input)?;
        let reshape = self.reshape.forward(&embedding)?;
        let linear = self.linear.forward(&reshape)?;
        let softmax = self.softmax.forward(&linear)?;
        Ok(softmax)
    }

    fn accelerator(&self) -> Rc<Accelerator> {
        self.embedding.accelerator()
    }

    fn tape(&self) -> Rc<RefCell<Tape>> {
        self.embedding.tape()
    }
}

pub fn architecture() -> Vec<DifferentiableModuleConfig> {
    vec![
        DifferentiableModuleConfig::Embedding(EmbeddingConfig {
            num_embeddings: 256,
            embedding_dim: 384,
        }),
        DifferentiableModuleConfig::Reshape(ReshapeConfig {
            input_rows: 32,
            input_cols: 384,
            output_rows: 1,
            output_cols: 32 * 384,
        }),
    ]
}
