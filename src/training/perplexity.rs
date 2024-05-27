use crate::{Error, Tensor};

/// https://en.wikipedia.org/wiki/Perplexity
pub fn get_perplexity(tensor: &Tensor, row: usize) -> Result<f32, Error> {
    let probabilities = tensor.get_values()?;
    let mut pp = 1.0;
    let cols = tensor.cols();
    for col in 0..cols {
        let p_x = probabilities[tensor.index(row, col)];
        pp *= p_x.powf(-p_x);
    }
    Ok(pp)
}
