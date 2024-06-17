use crate::{new_tensor, OperatorAttributes};
use crate::{
    optimization_instruction, tensor::Error, Device, Instruction, OpCode, OptimizerTrait,
    TensorWithGrad,
};

/// Adam: A Method for Stochastic Optimization
/// https://arxiv.org/abs/1412.6980
pub struct Adam {
    alpha: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
}

impl Adam {
    pub fn new(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            alpha: learning_rate,
            beta1,
            beta2,
            epsilon,
        }
    }
}

impl OptimizerTrait for Adam {
    fn optimize(
        &self,
        device: &Device,
        tensors: &[TensorWithGrad],
    ) -> Result<Vec<Instruction>, Error> {
        let alpha = self.alpha;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let epsilon = self.epsilon;

        let alpha_rate_tensor = new_tensor!(device, 1, 1, vec![alpha])?;
        let one_minus_beta1 = new_tensor!(device, 1, 1, vec![1.0 - beta1])?;
        let beta1_tensor = new_tensor!(device, 1, 1, vec![beta1])?;
        let one_minus_beta2 = new_tensor!(device, 1, 1, vec![1.0 - beta2])?;
        let beta2_tensor = new_tensor!(device, 1, 1, vec![beta2])?;
        let epsilon_tensor = new_tensor!(device, 1, 1, vec![epsilon])?;
        let zero = new_tensor!(device, 1, 1, vec![0.0])?;
        let f32_max = new_tensor!(device, 1, 1, vec![f32::MAX])?;

        let mut instructions = vec![];

        for optimizable_tensor in tensors {
            let theta = &optimizable_tensor.tensor();
            let g = &optimizable_tensor.gradient();
            debug_assert_eq!(*g.size(), *theta.size());

            let m = new_tensor!(device, theta.rows(), theta.cols(), vec![0.0; theta.len()])?;
            let v = new_tensor!(device, theta.rows(), theta.cols(), vec![0.0; theta.len()])?;

            let tmp1 = new_tensor!(device, theta.rows(), theta.cols(), vec![0.0; theta.len()])?;
            let tmp2 = new_tensor!(device, theta.rows(), theta.cols(), vec![0.0; theta.len()])?;

            // Update 1st moment
            // m = beta1 * m + (1 - beta1) * g
            instructions.push(optimization_instruction!(
                OpCode::ScalarMul,
                OperatorAttributes::None,
                &[&beta1_tensor, &m],
                &[&tmp1],
            ));
            instructions.push(optimization_instruction!(
                OpCode::ScalarMul,
                OperatorAttributes::None,
                &[&one_minus_beta1, &g],
                &[&tmp2],
            ));
            instructions.push(optimization_instruction!(
                OpCode::Add,
                OperatorAttributes::None,
                &[&tmp1, &tmp2],
                &[&m],
            ));

            // Update 2nd moment
            // v = beta2 * v + (1 - beta2) * g**2
            instructions.push(optimization_instruction!(
                OpCode::ScalarMul,
                OperatorAttributes::None,
                &[&beta2_tensor, &v],
                &[&tmp1],
            ));
            instructions.push(optimization_instruction!(
                OpCode::ScalarMul,
                OperatorAttributes::None,
                &[&g, &g],
                &[&tmp2],
            ));
            instructions.push(optimization_instruction!(
                OpCode::ScalarMul,
                OperatorAttributes::None,
                &[&one_minus_beta2, &tmp2],
                &[&tmp2],
            ));
            instructions.push(optimization_instruction!(
                OpCode::Add,
                OperatorAttributes::None,
                &[&tmp1, &tmp2],
                &[&v],
            ));

            // Correct bias in 1st and 2nd moments
            // TODO t should be in 0..num_iterations
            let t = 10;
            // m_hat = m / (1 - beta1**t)
            let m_hat = new_tensor!(device, theta.rows(), theta.cols(), vec![0.0; theta.len()])?;
            let m_multiplier = new_tensor!(device, 1, 1, vec![1.0 / (1.0 - beta1.powi(t))])?;
            instructions.push(optimization_instruction!(
                OpCode::ScalarMul,
                OperatorAttributes::None,
                &[&m_multiplier, &m],
                &[&m_hat],
            ));
            // v_hat = v / (1 - beta2**t)
            let v_hat = new_tensor!(device, theta.rows(), theta.cols(), vec![0.0; theta.len()])?;
            let v_multiplier = new_tensor!(device, 1, 1, vec![1.0 / (1.0 - beta2.powi(t))])?;
            instructions.push(optimization_instruction!(
                OpCode::ScalarMul,
                OperatorAttributes::None,
                &[&v_multiplier, &v],
                &[&v_hat],
            ));

            // Update parameters with adaptive learning rate
            // theta = theta - alpha * m_hat / (sqrt(v_hat) + epsilon)
            // Clip is used to remove negative values in v_hat.
            instructions.push(optimization_instruction!(
                OpCode::Clip,
                OperatorAttributes::None,
                &[&zero, &f32_max, &v_hat],
                &[&tmp1],
            ));
            instructions.push(optimization_instruction!(
                OpCode::Sqrt,
                OperatorAttributes::None,
                &[&tmp1],
                &[&tmp1],
            ));
            instructions.push(optimization_instruction!(
                OpCode::ScalarAdd,
                OperatorAttributes::None,
                &[&epsilon_tensor, &tmp1], // TODO use tmp1
                &[&tmp1],
            ));
            instructions.push(optimization_instruction!(
                OpCode::Div,
                OperatorAttributes::None,
                &[&m_hat, &tmp1],
                &[&tmp1],
            ));

            // ClipNorm is not in the adam paper. but +inf is reached is this is not done.
            // It's basically like clipping the gradient.
            instructions.push(optimization_instruction!(
                OpCode::ClipNorm,
                OperatorAttributes::None,
                &[&tmp1],
                &[&tmp1],
            ));

            instructions.push(optimization_instruction!(
                OpCode::ScalarMul,
                OperatorAttributes::None,
                &[&alpha_rate_tensor, &tmp1],
                &[&tmp1],
            ));
            instructions.push(optimization_instruction!(
                OpCode::Sub,
                OperatorAttributes::None,
                &[&theta, &tmp1],
                &[&theta],
            ));
        }
        Ok(instructions)
    }
}
