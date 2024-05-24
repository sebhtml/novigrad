use crate::{DeviceInterface, Error, Tensor};



pub struct RowMax {
    
}

impl RowMax {
    pub fn execute(inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        let input = inputs[0];
        let output = outputs[0];
        let device = input.device();
        device.row_max(input, output)
    }
}