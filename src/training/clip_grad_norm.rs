use std::ops::Deref;

use crate::{
    instruction, new_tensor,
    opcode::OpCode,
    tensor::{Error, Tensor},
    Category, Device, Instruction, OperatorAttributes,
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
        instructions.push(instruction!(
            OpCode::Dot,
            OperatorAttributes::None,
            &[&g, &g,],
            &[&g_dot],
            Category::Optimization,
        ));
        instructions.push(instruction!(
            OpCode::Add,
            OperatorAttributes::None,
            &[&l2_norm, &g_dot],
            &[&l2_norm],
            Category::Optimization,
        ));
    }
    instructions.push(instruction!(
        OpCode::Sqrt,
        OperatorAttributes::None,
        &[&l2_norm],
        &[&l2_norm],
        Category::Optimization,
    ));
    instructions.push(instruction!(
        OpCode::Div,
        OperatorAttributes::None,
        &[&one, &l2_norm],
        &[&alpha],
        Category::Optimization,
    ));
    instructions.push(instruction!(
        OpCode::Min,
        OperatorAttributes::None,
        &[&one, &alpha],
        &[&alpha],
        Category::Optimization,
    ));
    for g in gradient.iter() {
        instructions.push(instruction!(
            OpCode::ScalarMul,
            OperatorAttributes::None,
            &[&alpha, &g],
            &[&g],
            Category::Optimization,
        ));
    }
    Ok(instructions)
}
