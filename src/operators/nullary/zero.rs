use crate::{Error, Operator, Tensor};

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

    fn forward(&self, _inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), Error> {
        for output in outputs {
            output.zero()?;
        }
        Ok(())
    }
}
