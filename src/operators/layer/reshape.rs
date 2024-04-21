use std::rc::Rc;

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
    fn backward(
        &self,
        accelerator: &Accelerator,
        _error_working_memory: &mut DeltaWorkingMemory,
        _inputs: &Vec<Rc<Tensor>>,
        _output: &Rc<Tensor>,
        back_propagated_delta: &mut Tensor,
        layer_delta: &mut Tensor,
    ) -> Result<(Tensor, Vec<Gradient>), Error> {
        layer_delta.assign(accelerator, back_propagated_delta);
        layer_delta.reshape(self.input_rows, self.input_cols)?;

        back_propagated_delta.assign(accelerator, layer_delta);

        Ok((back_propagated_delta.clone(), vec![]))
    }

    fn forward(
        &self,
        accelerator: &Accelerator,
        inputs: &Vec<Rc<Tensor>>,
    ) -> Result<Rc<Tensor>, Error> {
        debug_assert_eq!(inputs.len(), 1);
        let input = &inputs[0];
        debug_assert_eq!(input.rows(), self.input_rows);
        debug_assert_eq!(input.cols(), self.input_cols);
        let mut output = Tensor::default();
        output.assign(accelerator, input);
        output.reshape(self.output_rows, self.output_cols)?;
        Ok(Rc::new(output))
    }

    fn name(&self) -> &str {
        "Reshape"
    }
}
