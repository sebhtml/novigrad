use crate::{
    Accelerator, DifferentiableModule, DifferentiableModuleConfig, EmbeddingConfig, Error, Forward,
    FullDifferentiableModuleConfig, ReshapeConfig, Session, Tape, Tensor,
};
use std::borrow::Borrow;
use std::{cell::RefCell, rc::Rc};

pub struct Architecture {
    embedding: DifferentiableModule,
    linear_0: DifferentiableModule,
    sigmoid_0: DifferentiableModule,
    reshape: DifferentiableModule,
    linear_1: DifferentiableModule,
    sigmoid_1: DifferentiableModule,
    linear_2: DifferentiableModule,
    softmax: DifferentiableModule,
}

impl Default for Architecture {
    fn default() -> Self {
        let session = Session::default();
        let accelerator = session.accelerator();
        let tape = session.tape();
        let configs = architecture();
        let mut iterator = configs.iter().peekable();
        let linear_0 = session.linear(16, 32, 6);
        let linear_1 = session.linear(32, 6 * 16, 1);
        let linear_2 = session.linear(16, 32, 1);
        let softmax = session.softmax(true);
        Self {
            embedding: FullDifferentiableModuleConfig {
                accelerator: &accelerator,
                tape: &tape,
                config: iterator.next().unwrap(),
            }
            .borrow()
            .into(),
            linear_0,
            sigmoid_0: FullDifferentiableModuleConfig {
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
            linear_1,
            sigmoid_1: FullDifferentiableModuleConfig {
                accelerator: &accelerator,
                tape: &tape,
                config: iterator.next().unwrap(),
            }
            .borrow()
            .into(),
            linear_2,
            softmax,
        }
    }
}

impl Forward for Architecture {
    fn forward(&mut self, layer_input: &Tensor) -> Result<Tensor, Error> {
        let embedding = self.embedding.forward(layer_input)?;
        let linear_0 = self.linear_0.forward(&embedding)?;
        let sigmoid_0 = self.sigmoid_0.forward(&linear_0)?;
        let reshape = self.reshape.forward(&sigmoid_0)?;
        let linear_1 = self.linear_1.forward(&reshape)?;
        let sigmoid_1 = self.sigmoid_1.forward(&linear_1)?;
        let linear_2 = self.linear_2.forward(&sigmoid_1)?;
        let softmax = self.softmax.forward(&linear_2)?;
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
            num_embeddings: 16,
            embedding_dim: 32,
        }),
        DifferentiableModuleConfig::Sigmoid(Default::default()),
        DifferentiableModuleConfig::Reshape(ReshapeConfig {
            input_rows: 6,
            input_cols: 16,
            output_rows: 1,
            output_cols: 6 * 16,
        }),
        DifferentiableModuleConfig::Sigmoid(Default::default()),
    ]
}
