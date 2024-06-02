use crate::{tensor::Tensor, Device, Mask, UnaryOperator};

#[test]
fn forward() {
    let device = Device::default();
    let rows = 3;
    let cols = 3;
    let input = device
        .tensor_with_grad(rows, cols, vec![1.0; rows * cols], &[], false, false)
        .unwrap();
    let mask = Mask::try_new(&device, rows, cols).unwrap();

    let output = mask.forward(&input).unwrap();
    output.forward().unwrap();

    let actual: &Tensor = &output.tensor().read().unwrap();

    // A position i is allowed to attend to a position j if and only if i > j.
    // This means that a position can attend to previous positions, but not itself or future positions.++
    for i in 0..rows {
        for j in 0..cols {
            let expected_value = if i > j { 1.0 } else { 0.0 };
            assert_eq!(
                expected_value,
                actual.get_values().unwrap()[actual.index(i, j)],
                "actual {} i {}, j {},",
                actual,
                i,
                j,
            );
        }
    }
}
