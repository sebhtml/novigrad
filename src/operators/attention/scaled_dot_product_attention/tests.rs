use std::ops::Deref;

use crate::{Device, ScaledDotProductAttention, TensorF32, TernaryOperator};

#[test]
fn forward() {
    let device = Device::default();
    let rows = 3;
    let cols = 3;
    let mask = true;
    let input = device.tensor(rows, cols, vec![1.0; rows * cols], &[], false, false);
    let attention = ScaledDotProductAttention::try_new(&device, rows, cols, mask).unwrap();

    let output = attention.forward(&input, &input, &input).unwrap();
    output.forward().unwrap();

    let actual: &TensorF32 = &output.tensor().deref().borrow();

    let actual_values = actual.get_values().unwrap();
    for actual_value in actual_values {
        assert!(actual_value.is_finite());
    }
}
