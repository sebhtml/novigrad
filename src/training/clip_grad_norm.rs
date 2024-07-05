use std::ops::Deref;

use crate::{
    new_tensor,
    opcode::OpCode,
    optimization_instruction,
    tensor::{Error, Tensor},
    Device, Instruction, OperatorAttributes,
};

pub fn clip_grad_norm(
    device: &Device,
    gradient: &[impl Deref<Target = Tensor>],
) -> Result<Vec<Instruction>, Error> {
    let one = new_tensor!(device, 1, 1, vec![1.0])?;
    let alpha = new_tensor!(device, 1, 1, vec![0.0])?;
    let l2_norm = new_tensor!(device, 1, 1, vec![0.0])?;
    let g_dot = new_tensor!(device, 1, 1, vec![0.0])?;
    let mut instructions = vec![];
    for g in gradient.iter() {
        instructions.push(optimization_instruction!(
            OpCode::Dot,
            OperatorAttributes::None,
            &[&g, &g,],
            &[&g_dot],
        ));
        instructions.push(optimization_instruction!(
            OpCode::Add,
            OperatorAttributes::None,
            &[&l2_norm, &g_dot],
            &[&l2_norm],
        ));
    }
    instructions.push(optimization_instruction!(
        OpCode::Sqrt,
        OperatorAttributes::None,
        &[&l2_norm],
        &[&l2_norm],
    ));
    instructions.push(optimization_instruction!(
        OpCode::Div,
        OperatorAttributes::None,
        &[&one, &l2_norm],
        &[&alpha],
    ));
    instructions.push(optimization_instruction!(
        OpCode::Min,
        OperatorAttributes::None,
        &[&one, &alpha],
        &[&alpha],
    ));
    for g in gradient.iter() {
        instructions.push(optimization_instruction!(
            OpCode::ScalarMul,
            OperatorAttributes::None,
            &[&alpha, &g],
            &[&g],
        ));
    }
    Ok(instructions)
}
