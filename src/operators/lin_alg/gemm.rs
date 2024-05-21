use crate::{devices::Device, Error, GenericTensor};

pub struct Gemm {}

impl Gemm {
    pub fn new(_device: &Device) -> Self {
        Self {}
    }

    pub fn execute(
        trans_a: bool,
        trans_b: bool,
        trans_result: bool,
        inputs: &[&crate::GenericTensor],
        outputs: &[&crate::GenericTensor],
    ) -> Result<(), Error> {
        debug_assert_eq!(inputs.len(), 2);
        debug_assert_eq!(outputs.len(), 1);
        let input = inputs[0];
        let weights = inputs[1];
        let biases = outputs[0];
        let a = input;
        let b = weights;
        let c = biases;
        let transa = trans_a;
        let transb = trans_b;
        let transpose_result = trans_result;
        GenericTensor::gemm(transa, transb, 1.0, a, b, 1.0, c, transpose_result)
    }
}
