use crate::{
    instruction, new_tensor, opcode::OpCode, tensor::Error, Category, Device, Instruction,
    OperatorAttributes, TensorWithGrad,
};

/// See:
/// Adam: A Method for Stochastic Optimization
/// https://arxiv.org/abs/1412.6980
///
/// See:
/// On the Convergence of Adam and Beyond
/// https://arxiv.org/abs/1904.09237
///
/// See:
/// A Theory on Adam Instability in Large-Scale Machine Learning
/// https://arxiv.org/pdf/2304.09871
///
/// See:
/// Full Parameter Fine-tuning for Large Language Models with Limited Resources
/// https://arxiv.org/pdf/2306.09782
///
/// See:
/// Decoupled Weight Decay Regularization
/// https://arxiv.org/abs/1711.05101
pub fn optimize(
    device: &Device,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    is_adam_w: bool,
    tensors: &[TensorWithGrad],
) -> Result<Vec<Instruction>, Error> {
    let mut instructions = vec![];
    let one = new_tensor!(device, 1, 1, vec![1.0])?;
    let t = new_tensor!(device, 1, 1, vec![0.0])?;

    instructions.push(instruction!(
        OpCode::Add,
        OperatorAttributes::None,
        &[&one, &t],
        &[&t],
        Category::Optimization,
    ));

    let adam_w_remaining_weight_after_decay = 1.0 - learning_rate * weight_decay;
    let adam_w_remaining_weight_after_decay =
        new_tensor!(device, 1, 1, vec![adam_w_remaining_weight_after_decay])?;

    let learning_rate = new_tensor!(device, 1, 1, vec![learning_rate])?;
    let one_minus_beta1 = new_tensor!(device, 1, 1, vec![1.0 - beta1])?;
    let beta1 = new_tensor!(device, 1, 1, vec![beta1])?;
    let one_minus_beta2 = new_tensor!(device, 1, 1, vec![1.0 - beta2])?;
    let beta2 = new_tensor!(device, 1, 1, vec![beta2])?;
    let epsilon = new_tensor!(device, 1, 1, vec![epsilon])?;
    let f32_max = new_tensor!(device, 1, 1, vec![f32::MAX])?;

    for optimizable_tensor in tensors {
        let theta = &optimizable_tensor.tensor();

        if is_adam_w && weight_decay != 0.0 {
            instructions.push(instruction!(
                OpCode::ScalarMul,
                OperatorAttributes::None,
                &[&adam_w_remaining_weight_after_decay, &theta],
                &[&theta],
                Category::Optimization,
            ));
        }

        let g = &optimizable_tensor.gradient();
        debug_assert_eq!(*g.size(), *theta.size());

        // m_0
        let m = new_tensor!(device, theta.rows(), theta.cols(), vec![0.0; theta.len()])?;
        // v_0
        let v = new_tensor!(device, theta.rows(), theta.cols(), vec![0.0; theta.len()])?;

        let tmp1 = new_tensor!(device, theta.rows(), theta.cols(), vec![0.0; theta.len()])?;
        let tmp2 = new_tensor!(device, theta.rows(), theta.cols(), vec![0.0; theta.len()])?;

        // Update 1st moment
        // m = beta1 * m + (1 - beta1) * g
        instructions.push(instruction!(
            OpCode::ScalarMul,
            OperatorAttributes::None,
            &[&beta1, &m],
            &[&tmp1],
            Category::Optimization,
        ));
        instructions.push(instruction!(
            OpCode::ScalarMul,
            OperatorAttributes::None,
            &[&one_minus_beta1, &g],
            &[&tmp2],
            Category::Optimization,
        ));
        instructions.push(instruction!(
            OpCode::Add,
            OperatorAttributes::None,
            &[&tmp1, &tmp2],
            &[&m],
            Category::Optimization,
        ));

        // Update 2nd moment
        // v = beta2 * v + (1 - beta2) * g**2
        instructions.push(instruction!(
            OpCode::ScalarMul,
            OperatorAttributes::None,
            &[&beta2, &v],
            &[&tmp1],
            Category::Optimization,
        ));
        instructions.push(instruction!(
            OpCode::Mul,
            OperatorAttributes::None,
            &[&g, &g],
            &[&tmp2],
            Category::Optimization,
        ));
        instructions.push(instruction!(
            OpCode::ScalarMul,
            OperatorAttributes::None,
            &[&one_minus_beta2, &tmp2],
            &[&tmp2],
            Category::Optimization,
        ));
        instructions.push(instruction!(
            OpCode::Add,
            OperatorAttributes::None,
            &[&tmp1, &tmp2],
            &[&v],
            Category::Optimization,
        ));

        // Correct bias in 1st and 2nd moments

        // m_hat = m / (1 - beta1**t)

        // m_multiplier = 1 / (1 - beta1**t)
        let m_multiplier = new_tensor!(device, 1, 1, vec![0.0])?;

        instructions.push(instruction!(
            OpCode::Pow,
            OperatorAttributes::None,
            &[&beta1, &t],
            &[&m_multiplier],
            Category::Optimization,
        ));
        instructions.push(instruction!(
            OpCode::Sub,
            OperatorAttributes::None,
            &[&one, &m_multiplier],
            &[&m_multiplier],
            Category::Optimization,
        ));
        instructions.push(instruction!(
            OpCode::Clip,
            OperatorAttributes::None,
            &[&epsilon, &f32_max, &m_multiplier],
            &[&m_multiplier],
            Category::Optimization,
        ));
        instructions.push(instruction!(
            OpCode::Div,
            OperatorAttributes::None,
            &[&one, &m_multiplier],
            &[&m_multiplier],
            Category::Optimization,
        ));

        let m_hat = new_tensor!(device, theta.rows(), theta.cols(), vec![0.0; theta.len()])?;

        // m_hatw
        instructions.push(instruction!(
            OpCode::ScalarMul,
            OperatorAttributes::None,
            &[&m_multiplier, &m],
            &[&m_hat],
            Category::Optimization,
        ));

        // v_hat = v / (1 - beta2**t)

        // v_multiplier = 1 / (1 - beta2**t)
        let v_multiplier = new_tensor!(device, 1, 1, vec![0.0])?;

        instructions.push(instruction!(
            OpCode::Pow,
            OperatorAttributes::None,
            &[&beta2, &t],
            &[&v_multiplier],
            Category::Optimization,
        ));
        instructions.push(instruction!(
            OpCode::Sub,
            OperatorAttributes::None,
            &[&one, &v_multiplier],
            &[&v_multiplier],
            Category::Optimization,
        ));
        instructions.push(instruction!(
            OpCode::Clip,
            OperatorAttributes::None,
            &[&epsilon, &f32_max, &v_multiplier],
            &[&v_multiplier],
            Category::Optimization,
        ));
        instructions.push(instruction!(
            OpCode::Div,
            OperatorAttributes::None,
            &[&one, &v_multiplier],
            &[&v_multiplier],
            Category::Optimization,
        ));

        let v_hat = new_tensor!(device, theta.rows(), theta.cols(), vec![0.0; theta.len()])?;

        // v_hat
        instructions.push(instruction!(
            OpCode::ScalarMul,
            OperatorAttributes::None,
            &[&v_multiplier, &v],
            &[&v_hat],
            Category::Optimization,
        ));

        // Update parameters with adaptive learning rate
        // theta = theta - alpha * m_hat / (sqrt(v_hat) + epsilon)

        instructions.push(instruction!(
            OpCode::Sqrt,
            OperatorAttributes::None,
            &[&tmp1],
            &[&tmp1],
            Category::Optimization,
        ));
        instructions.push(instruction!(
            OpCode::ScalarAdd,
            OperatorAttributes::None,
            &[&epsilon, &tmp1],
            &[&tmp1],
            Category::Optimization,
        ));
        instructions.push(instruction!(
            OpCode::Div,
            OperatorAttributes::None,
            &[&m_hat, &tmp1],
            &[&tmp1],
            Category::Optimization,
        ));

        // ClipNorm is not in the adam paper. but +inf is reached is this is not done.
        // It's basically like clipping the gradient.
        // TODO try to remove this ClipNorm thing.
        instructions.push(instruction!(
            OpCode::ClipNorm,
            OperatorAttributes::None,
            &[&tmp1],
            &[&tmp1],
            Category::Optimization,
        ));

        instructions.push(instruction!(
            OpCode::ScalarMul,
            OperatorAttributes::None,
            &[&learning_rate, &tmp1],
            &[&tmp1],
            Category::Optimization,
        ));
        instructions.push(instruction!(
            OpCode::Sub,
            OperatorAttributes::None,
            &[&theta, &tmp1],
            &[&theta],
            Category::Optimization,
        ));
    }
    Ok(instructions)
}
