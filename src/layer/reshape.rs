use crate::{Error, Layer, LayerType, Tensor, TensorTrait};

pub struct Reshape {
    input_rows: usize,
    input_cols: usize,
    output_rows: usize,
    output_cols: usize,
    forward_tensor: Tensor,
}

impl Reshape {
    pub fn new(
        input_rows: usize,
        input_cols: usize,
        output_rows: usize,
        output_cols: usize,
    ) -> Self {
        Self {
            input_rows,
            input_cols,
            output_rows,
            output_cols,
            forward_tensor: Default::default(),
        }
    }
}

impl Layer for Reshape {
    fn plan_change(
        &mut self,
        _learning_rate: f32,
        _previous_activation: &Tensor,
        _layer_delta: &Tensor,
    ) {
    }

    fn commit_change(&mut self) -> Result<(), Error> {
        Ok(())
    }

    fn forward(&mut self, input: &Tensor) -> Result<(), Error> {
        debug_assert_eq!(input.rows(), self.input_rows);
        debug_assert_eq!(input.cols(), self.input_cols);
        self.forward_tensor.assign(input);
        self.forward_tensor
            .reshape(self.output_rows, self.output_cols)
    }

    fn get_activation_tensor<'a>(&'a self) -> &'a Tensor {
        &self.forward_tensor
    }

    fn backward(&self, layer_delta: &Tensor, output_diff: &mut Tensor) {
        output_diff.assign(layer_delta);
    }

    fn get_layer_delta(
        &self,
        _working_memory: &mut crate::DeltaWorkingMemory,
        next_layer: Option<&LayerType>,
        next_layer_delta: &Tensor,
        _using_softmax_and_cross_entropy_loss: bool,
        layer_delta: &mut Tensor,
    ) {
        match next_layer {
            None => panic!("Not implemented"),
            Some(next_layer) => {
                // Hidden layer
                next_layer.backward(next_layer_delta, layer_delta);
                let op_result = layer_delta.reshape(self.input_rows, self.input_cols);
                op_result.expect("Ok");
            }
        }
    }
}

pub struct ReshapeConfig {
    pub input_rows: usize,
    pub input_cols: usize,
    pub output_rows: usize,
    pub output_cols: usize,
}

impl Into<Reshape> for &ReshapeConfig {
    fn into(self) -> Reshape {
        Reshape::new(
            self.input_rows,
            self.input_cols,
            self.output_rows,
            self.output_cols,
        )
    }
}
