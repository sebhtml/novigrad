use crate::{
    error,
    tensor::{Error, ErrorEnum},
    Add, BinaryOperator, Device, MatMul, TensorWithGrad, UnaryOperator,
};
use rand::{thread_rng, Rng};
use rand_distr::Normal;

pub enum WeightsInitialization {
    None,
    Kaiming,
    Xavier,
}

pub struct Linear {
    weights: TensorWithGrad,
    biases: TensorWithGrad,
    matmul: MatMul,
    add: Add,
}

fn kaiming_initialization(
    weights_rows: usize,
    _weights_cols: usize,
    weights: &mut Vec<f32>,
) -> Result<(), Error> {
    let mut rng = thread_rng();
    let mean = 0.0;
    let fan_in = weights_rows as f32;
    let stddev = (2.0 / fan_in).sqrt();
    let distribution =
        Normal::new(mean, stddev).map_err(|_| error!(ErrorEnum::UnsupportedOperation))?;

    for index in 0..weights.len() {
        weights[index] = rng.sample(distribution);
    }
    Ok(())
}

impl Linear {
    pub fn new(
        device: &Device,
        weights_rows: usize,
        weights_cols: usize,
        weights_initialization: WeightsInitialization,
        bias_rows: usize,
    ) -> Result<Self, Error> {
        let mut weights = Vec::new();
        weights.resize(weights_rows * weights_cols, 0.0);
        match weights_initialization {
            WeightsInitialization::None => {}
            WeightsInitialization::Kaiming => {
                kaiming_initialization(weights_rows, weights_cols, &mut weights)?;
            }
            WeightsInitialization::Xavier => todo!(),
        }

        let weights =
            device.tensor_with_grad(weights_rows, weights_cols, weights, &[], true, true)?;

        let biases_len = bias_rows * weights_rows;
        let biases = device.tensor_with_grad(
            bias_rows,
            weights_rows,
            vec![0.0; biases_len],
            &[],
            true,
            true,
        )?;

        let transb = true;
        let op = Self {
            weights,
            biases,
            matmul: MatMul::new(device, transb),
            add: Add::new(device),
        };
        Ok(op)
    }
}

impl UnaryOperator for Linear {
    fn forward(&self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
        let product = self.matmul.forward(input, &self.weights)?;
        let sum = self.add.forward(&product, &self.biases)?;
        Ok(sum)
    }
}
