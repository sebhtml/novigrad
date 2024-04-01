pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for idx in 0..a.len() {
        sum += a[idx] * b[idx];
    }
    sum
}
