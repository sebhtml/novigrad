use crate::{
    tensor::Error, Device, Linear, Model, TensorWithGrad, UnaryModel, UnaryOperator,
    WeightsInitialization,
};

pub struct PerceptronModel {
    linear: Linear,
}

impl UnaryModel for PerceptronModel {}

impl PerceptronModel {
    pub fn new(device: &Device) -> Result<Self, Error> {
        let linear = Linear::new(device, 1, 2, WeightsInitialization::Kaiming, 1)?;
        let model = Self { linear };
        Ok(model)
    }
}

impl UnaryOperator for PerceptronModel {
    fn forward(&self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
        self.linear.forward(input)
    }
}

impl Model for PerceptronModel {
    fn input_size(&self) -> Vec<usize> {
        vec![1, 2]
    }
    fn output_size(&self) -> Vec<usize> {
        vec![1, 1]
    }
}
