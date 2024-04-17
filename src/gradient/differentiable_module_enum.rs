use crate::{
    Accelerator, DeltaWorkingMemory, DifferentiableModuleTrait, Embedding, EmbeddingConfig, Error,
    Linear, LinearConfig, Reshape, ReshapeConfig, Sigmoid, SigmoidConfig, Softmax, Tensor,
};

pub enum DifferentiableModuleEnum {
    Embedding(Embedding),
    Linear(Linear),
    Reshape(Reshape),
    Sigmoid(Sigmoid),
    Softmax(Softmax),
}

impl DifferentiableModuleTrait for DifferentiableModuleEnum {
    fn compute_gradient(
        &mut self,
        accelerator: &Accelerator,
        layer_input: &Tensor,
        layer_output_delta: &Tensor,
    ) {
        match self {
            DifferentiableModuleEnum::Embedding(that) => {
                that.compute_gradient(accelerator, layer_input, layer_output_delta)
            }
            DifferentiableModuleEnum::Linear(that) => {
                that.compute_gradient(accelerator, layer_input, layer_output_delta)
            }
            DifferentiableModuleEnum::Reshape(that) => {
                that.compute_gradient(accelerator, layer_input, layer_output_delta)
            }
            DifferentiableModuleEnum::Sigmoid(that) => {
                that.compute_gradient(accelerator, layer_input, layer_output_delta)
            }
            DifferentiableModuleEnum::Softmax(that) => {
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
            DifferentiableModuleEnum::Embedding(that) => {
                that.commit_change(accelerator, learning_rate)
            }
            DifferentiableModuleEnum::Linear(that) => {
                that.commit_change(accelerator, learning_rate)
            }
            DifferentiableModuleEnum::Reshape(that) => {
                that.commit_change(accelerator, learning_rate)
            }
            DifferentiableModuleEnum::Sigmoid(that) => {
                that.commit_change(accelerator, learning_rate)
            }
            DifferentiableModuleEnum::Softmax(that) => {
                that.commit_change(accelerator, learning_rate)
            }
        }
    }

    fn forward(
        &mut self,
        accelerator: &Accelerator,
        input: &Tensor,
        output: &mut Tensor,
    ) -> Result<(), Error> {
        match self {
            DifferentiableModuleEnum::Embedding(that) => that.forward(accelerator, input, output),
            DifferentiableModuleEnum::Linear(that) => that.forward(accelerator, input, output),
            DifferentiableModuleEnum::Reshape(that) => that.forward(accelerator, input, output),
            DifferentiableModuleEnum::Sigmoid(that) => that.forward(accelerator, input, output),
            DifferentiableModuleEnum::Softmax(that) => that.forward(accelerator, input, output),
        }
    }

    fn backward(
        &self,
        accelerator: &Accelerator,
        layer_delta: &Tensor,
        previous_layer_delta: &mut Tensor,
    ) {
        match self {
            DifferentiableModuleEnum::Embedding(that) => {
                that.backward(accelerator, layer_delta, previous_layer_delta)
            }
            DifferentiableModuleEnum::Linear(that) => {
                that.backward(accelerator, layer_delta, previous_layer_delta)
            }
            DifferentiableModuleEnum::Reshape(that) => {
                that.backward(accelerator, layer_delta, previous_layer_delta)
            }
            DifferentiableModuleEnum::Sigmoid(that) => {
                that.backward(accelerator, layer_delta, previous_layer_delta)
            }
            DifferentiableModuleEnum::Softmax(that) => {
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
            DifferentiableModuleEnum::Embedding(that) => that.get_layer_output_delta(
                accelerator,
                working_memory,
                layer_input,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            DifferentiableModuleEnum::Linear(that) => that.get_layer_output_delta(
                accelerator,
                working_memory,
                layer_input,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            DifferentiableModuleEnum::Reshape(that) => that.get_layer_output_delta(
                accelerator,
                working_memory,
                layer_input,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            DifferentiableModuleEnum::Sigmoid(that) => that.get_layer_output_delta(
                accelerator,
                working_memory,
                layer_input,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            DifferentiableModuleEnum::Softmax(that) => that.get_layer_output_delta(
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

pub enum DifferentiableModuleConfig {
    Embedding(EmbeddingConfig),
    Linear(LinearConfig),
    Reshape(ReshapeConfig),
    Sigmoid(SigmoidConfig),
}

impl Into<DifferentiableModuleEnum> for &DifferentiableModuleConfig {
    fn into(self) -> DifferentiableModuleEnum {
        match self {
            DifferentiableModuleConfig::Embedding(config) => {
                DifferentiableModuleEnum::Embedding(config.into())
            }
            DifferentiableModuleConfig::Linear(config) => {
                DifferentiableModuleEnum::Linear(config.into())
            }
            DifferentiableModuleConfig::Reshape(config) => {
                DifferentiableModuleEnum::Reshape(config.into())
            }
            DifferentiableModuleConfig::Sigmoid(config) => {
                DifferentiableModuleEnum::Sigmoid(config.into())
            }
        }
    }
}
