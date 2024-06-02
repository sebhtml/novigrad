use crate::{tensor::Tensor, Device, ScaledDotProductAttention, TernaryOperator};

#[test]
fn forward() {
    let device = Device::default();
    let rows = 16;
    let cols = 384;
    let mask = true;
    let input = device
        .tensor_with_grad(rows, cols, vec![1.0; rows * cols], &[], false, false)
        .unwrap();
    let dropout_probability = 0.1;
    let attention =
        ScaledDotProductAttention::try_new(&device, rows, cols, mask, dropout_probability).unwrap();

    let output = attention.forward(&input, &input, &input).unwrap();
    output.forward().unwrap();

    let actual: &Tensor = &output.tensor().read().unwrap();

    let actual_values = actual.get_values().unwrap();
    for actual_value in actual_values {
        assert!(actual_value.is_finite());
    }
}
