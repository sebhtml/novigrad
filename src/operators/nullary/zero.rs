use crate::{Error, TensorF32};

pub struct Zero {}

impl Zero {
    pub fn execute(_inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        for output in outputs {
            debug_assert_eq!(false, output.is_nan()?);
            output.zero()?;
            debug_assert_eq!(false, output.is_nan()?);
        }
        Ok(())
    }
}
