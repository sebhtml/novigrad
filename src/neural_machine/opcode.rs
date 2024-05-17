use std::rc::Rc;

use crate::{Error, Gemm, Operator, TensorF32};

#[derive(Clone, Debug)]
pub enum OpCode {
    DynOperator(Rc<dyn Operator>),
    Gemm(bool, bool, bool),
}

impl Operator for OpCode {
    fn name(&self) -> &str {
        match self {
            OpCode::DynOperator(inner) => inner.name(),
            OpCode::Gemm(_, _, _) => "Gemm",
        }
    }

    fn forward(&self, inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        match self {
            OpCode::DynOperator(inner) => inner.forward(inputs, outputs),
            OpCode::Gemm(trans_a, trans_b, trans_result) => {
                Gemm::execute(*trans_a, *trans_b, *trans_result, inputs, outputs)
            }
        }
    }
}
