use crate::{
    Accelerator, CrossEntropyLoss, DeltaWorkingMemory, Embedding, Error, Linear, OperatorTrait,
    Reshape, ResidualSumOfSquares, Sigmoid, Softmax, Tensor,
};

pub enum OperatorEnum {
    Embedding(Embedding),
    Linear(Linear),
    Reshape(Reshape),
    Sigmoid(Sigmoid),
    Softmax(Softmax),
    ResidualSumOfSquares(ResidualSumOfSquares),
    CrossEntropyLoss(CrossEntropyLoss),
}

impl OperatorTrait for OperatorEnum {
    fn compute_gradient(
        &mut self,
        accelerator: &Accelerator,
        inputs: &Vec<Tensor>,
        layer_output_delta: &Tensor,
    ) {
        match self {
            OperatorEnum::Embedding(operator) => {
                operator.compute_gradient(accelerator, inputs, layer_output_delta)
            }
            OperatorEnum::Linear(operator) => {
                operator.compute_gradient(accelerator, inputs, layer_output_delta)
            }
            OperatorEnum::Reshape(operator) => {
                operator.compute_gradient(accelerator, inputs, layer_output_delta)
            }
            OperatorEnum::Sigmoid(operator) => {
                operator.compute_gradient(accelerator, inputs, layer_output_delta)
            }
            OperatorEnum::Softmax(operator) => {
                operator.compute_gradient(accelerator, inputs, layer_output_delta)
            }
            OperatorEnum::ResidualSumOfSquares(operator) => {
                operator.compute_gradient(accelerator, inputs, layer_output_delta)
            }
            OperatorEnum::CrossEntropyLoss(operator) => {
                operator.compute_gradient(accelerator, inputs, layer_output_delta)
            }
        }
    }

    fn commit_change(
        &mut self,
        accelerator: &Accelerator,
        learning_rate: f32,
    ) -> Result<(), Error> {
        match self {
            OperatorEnum::Embedding(operator) => operator.commit_change(accelerator, learning_rate),
            OperatorEnum::Linear(operator) => operator.commit_change(accelerator, learning_rate),
            OperatorEnum::Reshape(operator) => operator.commit_change(accelerator, learning_rate),
            OperatorEnum::Sigmoid(operator) => operator.commit_change(accelerator, learning_rate),
            OperatorEnum::Softmax(operator) => operator.commit_change(accelerator, learning_rate),
            OperatorEnum::ResidualSumOfSquares(operator) => {
                operator.commit_change(accelerator, learning_rate)
            }
            OperatorEnum::CrossEntropyLoss(operator) => {
                operator.commit_change(accelerator, learning_rate)
            }
        }
    }

    fn forward(
        &mut self,
        accelerator: &Accelerator,
        inputs: &Vec<Tensor>,
        output: &mut Tensor,
    ) -> Result<(), Error> {
        match self {
            OperatorEnum::Embedding(operator) => operator.forward(accelerator, inputs, output),
            OperatorEnum::Linear(operator) => operator.forward(accelerator, inputs, output),
            OperatorEnum::Reshape(operator) => operator.forward(accelerator, inputs, output),
            OperatorEnum::Sigmoid(operator) => operator.forward(accelerator, inputs, output),
            OperatorEnum::Softmax(operator) => operator.forward(accelerator, inputs, output),
            OperatorEnum::ResidualSumOfSquares(operator) => {
                operator.forward(accelerator, inputs, output)
            }
            OperatorEnum::CrossEntropyLoss(operator) => {
                operator.forward(accelerator, inputs, output)
            }
        }
    }

    fn backward(
        &self,
        inputs: &Vec<Tensor>,
        accelerator: &Accelerator,
        layer_delta: &Tensor,
        previous_layer_delta: &mut Tensor,
    ) {
        match self {
            OperatorEnum::Embedding(operator) => {
                operator.backward(inputs, accelerator, layer_delta, previous_layer_delta)
            }
            OperatorEnum::Linear(operator) => {
                operator.backward(inputs, accelerator, layer_delta, previous_layer_delta)
            }
            OperatorEnum::Reshape(operator) => {
                operator.backward(inputs, accelerator, layer_delta, previous_layer_delta)
            }
            OperatorEnum::Sigmoid(operator) => {
                operator.backward(inputs, accelerator, layer_delta, previous_layer_delta)
            }
            OperatorEnum::Softmax(operator) => {
                operator.backward(inputs, accelerator, layer_delta, previous_layer_delta)
            }
            OperatorEnum::ResidualSumOfSquares(operator) => {
                operator.backward(inputs, accelerator, layer_delta, previous_layer_delta)
            }
            OperatorEnum::CrossEntropyLoss(operator) => {
                operator.backward(inputs, accelerator, layer_delta, previous_layer_delta)
            }
        }
    }

    fn get_layer_output_delta(
        &self,
        accelerator: &Accelerator,
        working_memory: &mut DeltaWorkingMemory,
        inputs: &Vec<Tensor>,
        layer_output: &Tensor,
        back_propagated_delta: &Tensor,
        is_last_layer: bool,
        layer_delta: &mut Tensor,
    ) {
        match self {
            OperatorEnum::Embedding(operator) => operator.get_layer_output_delta(
                accelerator,
                working_memory,
                inputs,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            OperatorEnum::Linear(operator) => operator.get_layer_output_delta(
                accelerator,
                working_memory,
                inputs,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            OperatorEnum::Reshape(operator) => operator.get_layer_output_delta(
                accelerator,
                working_memory,
                inputs,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            OperatorEnum::Sigmoid(operator) => operator.get_layer_output_delta(
                accelerator,
                working_memory,
                inputs,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            OperatorEnum::Softmax(operator) => operator.get_layer_output_delta(
                accelerator,
                working_memory,
                inputs,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            OperatorEnum::ResidualSumOfSquares(operator) => operator.get_layer_output_delta(
                accelerator,
                working_memory,
                inputs,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
            OperatorEnum::CrossEntropyLoss(operator) => operator.get_layer_output_delta(
                accelerator,
                working_memory,
                inputs,
                layer_output,
                back_propagated_delta,
                is_last_layer,
                layer_delta,
            ),
        }
    }
}
