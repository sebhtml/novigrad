mod activation;
pub use activation::*;
mod view;
pub use view::*;
mod loss;
pub use loss::*;
mod lin_alg;
pub use lin_alg::*;
mod attention;
pub use attention::*;
mod reduce;
pub use reduce::*;
pub mod analysis;
pub mod opcode;
pub mod statistics;

use crate::{
    stream::DeviceStream,
    tensor::{Error, Tensor},
    Device, TensorWithGrad,
};

pub trait UnaryOperator {
    fn forward(&self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error>;
}

pub trait BinaryOperator {
    fn forward(
        &self,
        input_1: &TensorWithGrad,
        input_2: &TensorWithGrad,
    ) -> Result<TensorWithGrad, Error>;
}

pub trait TernaryOperator {
    fn forward(
        &self,
        input_1: &TensorWithGrad,
        input_2: &TensorWithGrad,
        input_3: &TensorWithGrad,
    ) -> Result<TensorWithGrad, Error>;
}

/// An n-ary function takes n arguments.
/// https://en.wikipedia.org/wiki/Arity
pub trait NaryOperator {
    fn forward(&self, inputs: &[&TensorWithGrad]) -> Result<TensorWithGrad, Error>;
}

pub trait ExecutableOperator {
    fn execute(
        attributes: &OperatorAttributes,
        inputs: &[&Tensor],
        outputs: &[&Tensor],
        device: &Device,
        device_stream: &DeviceStream,
    ) -> Result<(), Error>;
}

#[derive(Clone, Debug, Default)]
pub enum OperatorAttributes {
    #[default]
    None,
    ThreeBools(bool, bool, bool),
    String(String),
    Vec(Vec<usize>),
}
