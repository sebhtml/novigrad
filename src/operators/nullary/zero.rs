use crate::{Error, Operator, TensorF32};

pub struct Zero {}

impl Default for Zero {
    fn default() -> Self {
        Self {}
    }
}

impl Operator for Zero {
    fn name(&self) -> &str {
        "Zero"
    }

    fn forward(&self, _inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        for output in outputs {
            debug_assert_eq!(false, output.is_nan()?);
            output.zero()?;
            debug_assert_eq!(false, output.is_nan()?);
        }
        Ok(())
    }
}
