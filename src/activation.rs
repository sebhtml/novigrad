use std::f32::consts::E;

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + E.powf(-x))
}

pub fn sigmoid_derivative(x: f32) -> f32 {
    let sigmoid_x = sigmoid(x);
    sigmoid_x * (1.0 - sigmoid_x)
}
