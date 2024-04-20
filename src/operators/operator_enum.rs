use crate::{
    Accelerator, CrossEntropyLoss, DeltaWorkingMemory, Embedding, Error, Gradient, Linear,
    OperatorTrait, Reshape, ResidualSumOfSquares, Sigmoid, Softmax, Tensor,
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

impl Into<String> for &OperatorEnum {
    fn into(self) -> String {
        match self {
            OperatorEnum::Embedding(_) => "Embedding",
            OperatorEnum::Linear(_) => "Linear",
            OperatorEnum::Reshape(_) => "Reshape",
            OperatorEnum::Sigmoid(_) => "Sigmoid",
            OperatorEnum::Softmax(_) => "Softmax",
            OperatorEnum::ResidualSumOfSquares(_) => "ResidualSumOfSquares",
            OperatorEnum::CrossEntropyLoss(_) => "CrossEntropyLoss",
        }
        .into()
    }
}

impl OperatorEnum {
    pub fn name(&self) -> String {
        self.into()
    }
}

impl OperatorTrait for OperatorEnum {
    fn compute_gradients(
        &mut self,
        accelerator: &Accelerator,
        inputs: &Vec<Tensor>,
        layer_output_delta: &Tensor,
    ) -> Result<Vec<Gradient>, Error> {
        match self {
            OperatorEnum::Embedding(operator) => {
                operator.compute_gradients(accelerator, inputs, layer_output_delta)
            }
            OperatorEnum::Linear(operator) => {
                operator.compute_gradients(accelerator, inputs, layer_output_delta)
            }
            OperatorEnum::Reshape(operator) => {
                operator.compute_gradients(accelerator, inputs, layer_output_delta)
            }
            OperatorEnum::Sigmoid(operator) => {
                operator.compute_gradients(accelerator, inputs, layer_output_delta)
            }
            OperatorEnum::Softmax(operator) => {
                operator.compute_gradients(accelerator, inputs, layer_output_delta)
            }
            OperatorEnum::ResidualSumOfSquares(operator) => {
                operator.compute_gradients(accelerator, inputs, layer_output_delta)
            }
            OperatorEnum::CrossEntropyLoss(operator) => {
                operator.compute_gradients(accelerator, inputs, layer_output_delta)
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
        layer_delta: &mut Tensor,
    ) {
        match self {
            OperatorEnum::Embedding(operator) => operator.get_layer_output_delta(
                accelerator,
                working_memory,
                inputs,
                layer_output,
                back_propagated_delta,
                layer_delta,
            ),
            OperatorEnum::Linear(operator) => operator.get_layer_output_delta(
                accelerator,
                working_memory,
                inputs,
                layer_output,
                back_propagated_delta,
                layer_delta,
            ),
            OperatorEnum::Reshape(operator) => operator.get_layer_output_delta(
                accelerator,
                working_memory,
                inputs,
                layer_output,
                back_propagated_delta,
                layer_delta,
            ),
            OperatorEnum::Sigmoid(operator) => operator.get_layer_output_delta(
                accelerator,
                working_memory,
                inputs,
                layer_output,
                back_propagated_delta,
                layer_delta,
            ),
            OperatorEnum::Softmax(operator) => operator.get_layer_output_delta(
                accelerator,
                working_memory,
                inputs,
                layer_output,
                back_propagated_delta,
                layer_delta,
            ),
            OperatorEnum::ResidualSumOfSquares(operator) => operator.get_layer_output_delta(
                accelerator,
                working_memory,
                inputs,
                layer_output,
                back_propagated_delta,
                layer_delta,
            ),
            OperatorEnum::CrossEntropyLoss(operator) => operator.get_layer_output_delta(
                accelerator,
                working_memory,
                inputs,
                layer_output,
                back_propagated_delta,
                layer_delta,
            ),
        }
    }
}
