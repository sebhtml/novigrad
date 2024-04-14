use crate::{
    accelerator::Accelerator, DeltaWorkingMemory, Embedding, EmbeddingConfig, Error, Linear,
    LinearConfig, Reshape, ReshapeConfig, Sigmoid, SigmoidConfig, Softmax, SoftmaxConfig, Tensor,
};

pub struct DifferentiableTensor {
    pub tensor: Tensor,
    pub gradient: Tensor,
    pub has_gradient: bool,
}

impl DifferentiableTensor {
    pub fn new(tensor: Tensor) -> Self {
        Self {
            tensor,
            gradient: Default::default(),
            has_gradient: Default::default(),
        }
    }
    pub fn commit_change(&mut self, accelerator: &Accelerator, learning_rate: f32) {
        if !self.has_gradient {
            return;
        }

        let op_result = Tensor::saxpy(
            accelerator,
            -learning_rate,
            &self.gradient,
            &mut self.tensor,
        );
        op_result.expect("Ok");
        self.has_gradient = false;
    }
}

impl From<Tensor> for DifferentiableTensor {
    fn from(value: Tensor) -> Self {
        Self::new(value)
    }
}

pub trait DifferentiableModuleTrait {
    fn compute_gradient(
        &mut self,
        accelerator: &Accelerator,
        layer_input: &Tensor,
        layer_output_delta: &Tensor,
    );

    fn commit_change(&mut self, accelerator: &Accelerator, learning_rate: f32)
        -> Result<(), Error>;

    fn forward(
        &mut self,
        accelerator: &Accelerator,
        layer_input: &Tensor,
        layer_output: &mut Tensor,
    ) -> Result<(), Error>;

    // TODO backward should return Error
    fn backward(
        &self,
        accelerator: &Accelerator,
        layer_output_delta: &Tensor,
        previous_layer_output_delta: &mut Tensor,
    );

    // TODO get_layer_delta should return Error
    fn get_layer_output_delta(
        &self,
        accelerator: &Accelerator,
        working_memory: &mut DeltaWorkingMemory,
        layer_input: &Tensor,
        layer_output: &Tensor,
        back_propagated_layer_output_delta: &Tensor,
        is_last_layer: bool,
        layer_output_delta: &mut Tensor,
    );
}

pub enum DifferentiableModuleConfig {
    Embedding(EmbeddingConfig),
    Linear(LinearConfig),
    Reshape(ReshapeConfig),
    Sigmoid(SigmoidConfig),
    Softmax(SoftmaxConfig),
}

pub enum DifferentiableModule {
    Embedding(Embedding),
    Linear(Linear),
    Reshape(Reshape),
    Sigmoid(Sigmoid),
    Softmax(Softmax),
}

impl Into<DifferentiableModule> for &DifferentiableModuleConfig {
    fn into(self) -> DifferentiableModule {
        match self {
            DifferentiableModuleConfig::Embedding(config) => {
                DifferentiableModule::Embedding(config.into())
            }
            DifferentiableModuleConfig::Linear(config) => {
                DifferentiableModule::Linear(config.into())
            }
            DifferentiableModuleConfig::Reshape(config) => {
                DifferentiableModule::Reshape(config.into())
            }
            DifferentiableModuleConfig::Sigmoid(config) => {
                DifferentiableModule::Sigmoid(config.into())
            }
            DifferentiableModuleConfig::Softmax(config) => {
                DifferentiableModule::Softmax(config.into())
            }
        }
    }
}

impl DifferentiableModuleTrait for DifferentiableModule {
    fn compute_gradient(
        &mut self,
        accelerator: &Accelerator,
        layer_input: &Tensor,
        layer_output_delta: &Tensor,
    ) {
        match self {
            DifferentiableModule::Embedding(that) => {
                that.compute_gradient(accelerator, layer_input, layer_output_delta)
            }
            DifferentiableModule::Linear(that) => {
                that.compute_gradient(accelerator, layer_input, layer_output_delta)
            }
            DifferentiableModule::Reshape(that) => {
                that.compute_gradient(accelerator, layer_input, layer_output_delta)
            }
            DifferentiableModule::Sigmoid(that) => {
                that.compute_gradient(accelerator, layer_input, layer_output_delta)
            }
            DifferentiableModule::Softmax(that) => {
                that.compute_gradient(accelerator, layer_input, layer_output_delta)
            }
        }
    }

    fn commit_change(
        &mut self,
        accelerator: &Accelerator,
        learning_rate: f32,
    ) -> Result<(), Error> {
        match self {
            DifferentiableModule::Embedding(that) => that.commit_change(accelerator, learning_rate),
            DifferentiableModule::Linear(that) => that.commit_change(accelerator, learning_rate),
            DifferentiableModule::Reshape(that) => that.commit_change(accelerator, learning_rate),
            DifferentiableModule::Sigmoid(that) => that.commit_change(accelerator, learning_rate),
            DifferentiableModule::Softmax(that) => that.commit_change(accelerator, learning_rate),
        }
    }

    fn forward(
        &mut self,
        accelerator: &Accelerator,
        input: &Tensor,
        output: &mut Tensor,
    ) -> Result<(), Error> {
        match self {
            DifferentiableModule::Embedding(that) => that.forward(accelerator, input, output),
            DifferentiableModule::Linear(that) => that.forward(accelerator, input, output),
            DifferentiableModule::Reshape(that) => that.forward(accelerator, input, output),
            DifferentiableModule::Sigmoid(that) => that.forward(accelerator, input, output),
            DifferentiableModule::Softmax(that) => that.forward(accelerator, input, output),
        }
    }

    fn backward(
        &self,
        accelerator: &Accelerator,
        layer_delta: &Tensor,
        previous_layer_delta: &mut Tensor,
    ) {
        match self {
            DifferentiableModule::Embedding(that) => {
                that.backward(accelerator, layer_delta, previous_layer_delta)
            }
            DifferentiableModule::Linear(that) => {
                that.backward(accelerator, layer_delta, previous_layer_delta)
            }
            DifferentiableModule::Reshape(that) => {
                that.backward(accelerator, layer_delta, previous_layer_delta)
            }
            DifferentiableModule::Sigmoid(that) => {
                that.backward(accelerator, layer_delta, previous_layer_delta)
            }
            DifferentiableModule::Softmax(that) => {
                that.backward(accelerator, layer_delta, previous_layer_delta)
            }
        }
    }

    fn get_layer_output_delta(
        &self,
        accelerator: &Accelerator,
        working_memory: &mut DeltaWorkingMemory,
        layer_input: &Tensor,
        layer_output: &Tensor,
        back_propagated_delta: &Tensor,
        is_last_layer: bool,
        layer_delta: &mut Tensor,
    ) {
        match self {
            DifferentiableModule::Embedding(that) => that.get_layer_output_delta(
                accelerator,
                working_memory,
                layer_input,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            DifferentiableModule::Linear(that) => that.get_layer_output_delta(
                accelerator,
                working_memory,
                layer_input,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            DifferentiableModule::Reshape(that) => that.get_layer_output_delta(
                accelerator,
                working_memory,
                layer_input,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            DifferentiableModule::Sigmoid(that) => that.get_layer_output_delta(
                accelerator,
                working_memory,
                layer_input,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            DifferentiableModule::Softmax(that) => that.get_layer_output_delta(
                accelerator,
                working_memory,
                layer_input,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
        }
    }
}
