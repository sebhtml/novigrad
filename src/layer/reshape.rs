use crate::{DeltaWorkingMemory, Error, Layer, LayerType, Tensor, TensorTrait};

pub struct Reshape {
    input_rows: usize,
    input_cols: usize,
    output_rows: usize,
    output_cols: usize,
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

    fn forward(&mut self, input: &Tensor, output: &mut Tensor) -> Result<(), Error> {
        debug_assert_eq!(input.rows(), self.input_rows);
        debug_assert_eq!(input.cols(), self.input_cols);
        output.assign(input);
        output.reshape(self.output_rows, self.output_cols)
    }

    fn backward(&self, layer_delta: &Tensor, output_diff: &mut Tensor) {
        output_diff.assign(layer_delta);
    }

    fn get_layer_delta(
        &self,
        _working_memory: &mut DeltaWorkingMemory,
        _layer_input: &Tensor,
        _layer_output: &Tensor,
        back_propagated_delta: &Tensor,
        _is_last_layer: bool,
        layer_delta: &mut Tensor,
    ) {
        layer_delta.assign(back_propagated_delta);
        let op_result = layer_delta.reshape(self.input_rows, self.input_cols);
        op_result.expect("Ok");
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
