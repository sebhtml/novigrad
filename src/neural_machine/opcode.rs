use std::rc::Rc;

use crate::Operator;

#[derive(Clone, Debug)]
pub enum OpCode {
    DynOperator(Rc<dyn Operator>),
}

impl Operator for OpCode {
    fn name(&self) -> &str {
        match self {
            OpCode::DynOperator(inner) => inner.name(),
        }
    }

    fn forward(
        &self,
        inputs: &[&crate::TensorF32],
        outputs: &[&crate::TensorF32],
    ) -> Result<(), crate::Error> {
        match self {
            OpCode::DynOperator(inner) => inner.forward(inputs, outputs),
        }
    }
}
