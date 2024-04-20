use crate::{accelerator::Accelerator, DeltaWorkingMemory, Error, Gradient, OperatorTrait, Tensor};

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

impl OperatorTrait for Reshape {
    fn compute_gradients(
        &mut self,
        _accelerator: &Accelerator,
        _inputs: &Vec<Tensor>,
        _layer_output_delta: &Tensor,
    ) -> Result<Vec<Gradient>, Error> {
        Ok(vec![])
    }

    fn forward(
        &mut self,
        accelerator: &Accelerator,
        inputs: &Vec<Tensor>,
        output: &mut Tensor,
    ) -> Result<(), Error> {
        debug_assert_eq!(inputs.len(), 1);
        let input = &inputs[0];
        debug_assert_eq!(input.rows(), self.input_rows);
        debug_assert_eq!(input.cols(), self.input_cols);
        output.assign(accelerator, input);
        output.reshape(self.output_rows, self.output_cols)
    }

    fn backward(
        &self,
        _inputs: &Vec<Tensor>,
        accelerator: &Accelerator,
        layer_delta: &Tensor,
        output_diff: &mut Tensor,
    ) {
        output_diff.assign(accelerator, layer_delta);
    }

    fn get_layer_output_delta(
        &self,
        accelerator: &Accelerator,
        _working_memory: &mut DeltaWorkingMemory,
        _inputs: &Vec<Tensor>,
        _layer_output: &Tensor,
        back_propagated_delta: &Tensor,
        layer_delta: &mut Tensor,
    ) {
        layer_delta.assign(accelerator, back_propagated_delta);
        let op_result = layer_delta.reshape(self.input_rows, self.input_cols);
        op_result.expect("Ok");
    }

    fn name(&self) -> &str {
        "Reshape"
    }
}
