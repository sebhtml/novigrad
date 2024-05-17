use std::rc::Rc;

use crate::{
    Add, AddBackward, Clip, Error, Gemm, Identity, IdentityBackward, Mul, Operator, Reshape, Scale,
    ScaleBackward, Softmax, Sub, TensorF32, Zero,
};

#[derive(Clone, Debug)]
pub enum OpCode {
    DynOperator(Rc<dyn Operator>),
    Gemm(bool, bool, bool),
    Add,
    AddBackward,
    Scale(f32),
    ScaleBackward,
    Zero,
    Clip(f32),
    Mul,
    Identity,
    IdentityBackward,
    Softmax,
    Sub,
    Reshape(Vec<usize>),
}

impl Operator for OpCode {
    fn name(&self) -> &str {
        match self {
            OpCode::DynOperator(inner) => inner.name(),
            OpCode::Gemm(_, _, _) => "Gemm",
            OpCode::Add => "Add",
            OpCode::AddBackward => "AddBackward",
            OpCode::Scale(_) => "Scale",
            OpCode::ScaleBackward => "ScaleBackward",
            OpCode::Zero => "Zero",
            OpCode::Clip(_) => "Clip",
            OpCode::Mul => "Mul",
            OpCode::Identity => "Identity",
            OpCode::IdentityBackward => "IdentityBackward",
            OpCode::Softmax => "Softmax",
            OpCode::Sub => "Sub",
            OpCode::Reshape(_) => "Reshape",
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
            OpCode::ScaleBackward => ScaleBackward::execute(inputs, outputs),
            OpCode::Zero => Zero::execute(inputs, outputs),
            OpCode::Clip(clipped_norm) => Clip::execute(*clipped_norm, inputs, outputs),
            OpCode::Mul => Mul::execute(inputs, outputs),
            OpCode::Identity => Identity::execute(inputs, outputs),
            OpCode::IdentityBackward => IdentityBackward::execute(inputs, outputs),
            OpCode::Softmax => Softmax::execute(inputs, outputs),
            OpCode::Sub => Sub::execute(inputs, outputs),
            OpCode::Reshape(output_size) => Reshape::execute(output_size, inputs, outputs),
        }
    }
}
