use std::ops::Deref;

use crate::{
    new_tensor,
    tensor::{Error, Tensor},
    Device, Instruction,
};

pub fn clip_grad_norm(
    device: &Device,
    gradient: &[impl Deref<Target = Tensor>],
) -> Result<Vec<Instruction>, Error> {
    let l2_norm = new_tensor!(device, 1, 1, vec![0.0])?;
    let g_sum_squares = new_tensor!(device, 1, 1, vec![0.0])?;
    let mut instructions = vec![];
    for g in gradient.iter() {}
    Ok(instructions)
}
