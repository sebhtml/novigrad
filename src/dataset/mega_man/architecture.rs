use std::{borrow::Borrow, cell::RefCell, rc::Rc};

use crate::{
    Accelerator, DifferentiableModule, DifferentiableModuleConfig, EmbeddingConfig, Error, Forward,
    FullDifferentiableModuleConfig, LinearConfig, ReshapeConfig, SoftmaxConfig, Tape, Tensor,
};

pub struct Architecture {
    accelerator: Rc<Accelerator>,
    tape: Rc<RefCell<Tape>>,
    embedding: DifferentiableModule,
    reshape: DifferentiableModule,
    linear: DifferentiableModule,
    softmax: DifferentiableModule,
}

impl Default for Architecture {
    fn default() -> Self {
        let accelerator = Rc::new(Accelerator::default());
        let tape = Rc::new(RefCell::new(Tape::default()));
        let configs = architecture();
        let mut iterator = configs.iter().peekable();
        Self {
            accelerator: Default::default(),
            tape: Default::default(),
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
            linear: FullDifferentiableModuleConfig {
                accelerator: &accelerator,
                tape: &tape,
                config: iterator.next().unwrap(),
            }
            .borrow()
            .into(),
            softmax: FullDifferentiableModuleConfig {
                accelerator: &accelerator,
                tape: &tape,
                config: iterator.next().unwrap(),
            }
            .borrow()
            .into(),
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
        self.accelerator.clone()
    }

    fn tape(&self) -> Rc<RefCell<Tape>> {
        self.tape.clone()
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
        DifferentiableModuleConfig::Linear(LinearConfig {
            weights_rows: 256,
            weights_cols: 32 * 384,
            bias_rows: 1,
        }),
        DifferentiableModuleConfig::Softmax(SoftmaxConfig {
            using_cross_entropy_loss: true,
        }),
    ]
}
