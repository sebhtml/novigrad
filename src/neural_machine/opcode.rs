use std::rc::Rc;

use crate::{Add, AddBackward, Error, Gemm, Operator, Scale, TensorF32};

#[derive(Clone, Debug)]
pub enum OpCode {
    DynOperator(Rc<dyn Operator>),
    Gemm(bool, bool, bool),
    Add,
    AddBackward,
    Scale(f32),
}

impl Operator for OpCode {
    fn name(&self) -> &str {
        match self {
            OpCode::DynOperator(inner) => inner.name(),
            OpCode::Gemm(_, _, _) => "Gemm",
            OpCode::Add => "Add",
            OpCode::AddBackward => "AddBackward",
            OpCode::Scale(_) => "Scale",
        }
    }

    fn forward(&self, inputs: &[&TensorF32], outputs: &[&TensorF32]) -> Result<(), Error> {
        match self {
            OpCode::DynOperator(inner) => inner.forward(inputs, outputs),
            OpCode::Gemm(trans_a, trans_b, trans_result) => {
                Gemm::execute(*trans_a, *trans_b, *trans_result, inputs, outputs)
            }
            OpCode::Add => Add::execute(inputs, outputs),
            OpCode::AddBackward => AddBackward::execute(inputs, outputs),
            OpCode::Scale(alpha) => Scale::execute(*alpha, inputs, outputs),
        }
    }
}
