use crate::{
    common_adam::optimize, tensor::Error, Device, Instruction, OptimizerTrait, TensorWithGrad,
};

/// See:
/// Adam: A Method for Stochastic Optimization
/// https://arxiv.org/abs/1412.6980
///
/// See:
/// On the Convergence of Adam and Beyond
/// https://arxiv.org/abs/1904.09237
///
/// See:
/// A Theory on Adam Instability in Large-Scale Machine Learning
/// https://arxiv.org/pdf/2304.09871
///
/// See:
/// Full Parameter Fine-tuning for Large Language Models with Limited Resources
/// https://arxiv.org/pdf/2306.09782
pub struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
}

impl Adam {
    pub fn try_new(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> Result<Self, Error> {
        let adam = Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
        };
        Ok(adam)
    }
}

impl OptimizerTrait for Adam {
    fn optimize(
        &self,
        device: &Device,
        tensors: &[TensorWithGrad],
    ) -> Result<Vec<Instruction>, Error> {
        optimize(
            device,
            self.learning_rate,
            self.beta1,
            self.beta2,
            self.epsilon,
            self.weight_decay,
            tensors,
        )
    }
}
