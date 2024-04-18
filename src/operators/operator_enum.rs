use crate::{
    Accelerator, DeltaWorkingMemory, Embedding, Error, Linear, OperatorTrait, Reshape, Sigmoid,
    Softmax, Tensor,
};

pub enum OperatorEnum {
    Embedding(Embedding),
    Linear(Linear),
    Reshape(Reshape),
    Sigmoid(Sigmoid),
    Softmax(Softmax),
}

impl OperatorTrait for OperatorEnum {
    fn compute_gradient(
        &mut self,
        accelerator: &Accelerator,
        layer_input: &Tensor,
        layer_output_delta: &Tensor,
    ) {
        match self {
            OperatorEnum::Embedding(that) => {
                that.compute_gradient(accelerator, layer_input, layer_output_delta)
            }
            OperatorEnum::Linear(that) => {
                that.compute_gradient(accelerator, layer_input, layer_output_delta)
            }
            OperatorEnum::Reshape(that) => {
                that.compute_gradient(accelerator, layer_input, layer_output_delta)
            }
            OperatorEnum::Sigmoid(that) => {
                that.compute_gradient(accelerator, layer_input, layer_output_delta)
            }
            OperatorEnum::Softmax(that) => {
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
            OperatorEnum::Embedding(that) => that.commit_change(accelerator, learning_rate),
            OperatorEnum::Linear(that) => that.commit_change(accelerator, learning_rate),
            OperatorEnum::Reshape(that) => that.commit_change(accelerator, learning_rate),
            OperatorEnum::Sigmoid(that) => that.commit_change(accelerator, learning_rate),
            OperatorEnum::Softmax(that) => that.commit_change(accelerator, learning_rate),
        }
    }

    fn forward(
        &mut self,
        accelerator: &Accelerator,
        input: &Tensor,
        output: &mut Tensor,
    ) -> Result<(), Error> {
        match self {
            OperatorEnum::Embedding(that) => that.forward(accelerator, input, output),
            OperatorEnum::Linear(that) => that.forward(accelerator, input, output),
            OperatorEnum::Reshape(that) => that.forward(accelerator, input, output),
            OperatorEnum::Sigmoid(that) => that.forward(accelerator, input, output),
            OperatorEnum::Softmax(that) => that.forward(accelerator, input, output),
        }
    }

    fn backward(
        &self,
        accelerator: &Accelerator,
        layer_delta: &Tensor,
        previous_layer_delta: &mut Tensor,
    ) {
        match self {
            OperatorEnum::Embedding(that) => {
                that.backward(accelerator, layer_delta, previous_layer_delta)
            }
            OperatorEnum::Linear(that) => {
                that.backward(accelerator, layer_delta, previous_layer_delta)
            }
            OperatorEnum::Reshape(that) => {
                that.backward(accelerator, layer_delta, previous_layer_delta)
            }
            OperatorEnum::Sigmoid(that) => {
                that.backward(accelerator, layer_delta, previous_layer_delta)
            }
            OperatorEnum::Softmax(that) => {
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
            OperatorEnum::Embedding(that) => that.get_layer_output_delta(
                accelerator,
                working_memory,
                layer_input,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            OperatorEnum::Linear(that) => that.get_layer_output_delta(
                accelerator,
                working_memory,
                layer_input,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            OperatorEnum::Reshape(that) => that.get_layer_output_delta(
                accelerator,
                working_memory,
                layer_input,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            OperatorEnum::Sigmoid(that) => that.get_layer_output_delta(
                accelerator,
                working_memory,
                layer_input,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            OperatorEnum::Softmax(that) => that.get_layer_output_delta(
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
