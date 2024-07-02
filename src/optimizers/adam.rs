use crate::opcode::OpCode;
use crate::{new_tensor, OperatorAttributes};
use crate::{
    optimization_instruction, tensor::Error, Device, Instruction, OptimizerTrait, TensorWithGrad,
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
pub struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
}

impl Adam {
    pub fn try_new(
        _device: &Device,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        _weight_decay: f32,
    ) -> Result<Self, Error> {
        let adam = Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
        };
        Ok(adam)
    }
}

impl OptimizerTrait for Adam {
    fn optimize(
        &self,
        device: &Device,
        tensors: &[TensorWithGrad],
    ) -> Result<Vec<Instruction>, Error> {
        let mut instructions = vec![];
        let one = new_tensor!(device, 1, 1, vec![1.0])?;
        let t = new_tensor!(device, 1, 1, vec![0.0])?;

        instructions.push(optimization_instruction!(
            OpCode::Add,
            OperatorAttributes::None,
            &[&one, &t],
            &[&t],
        ));

        let learning_rate = self.learning_rate;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let epsilon = self.epsilon;

        let learning_rate = new_tensor!(device, 1, 1, vec![learning_rate])?;
        let one_minus_beta1 = new_tensor!(device, 1, 1, vec![1.0 - beta1])?;
        let beta1 = new_tensor!(device, 1, 1, vec![beta1])?;
        let one_minus_beta2 = new_tensor!(device, 1, 1, vec![1.0 - beta2])?;
        let beta2 = new_tensor!(device, 1, 1, vec![beta2])?;
        let epsilon = new_tensor!(device, 1, 1, vec![epsilon])?;
        let f32_max = new_tensor!(device, 1, 1, vec![f32::MAX])?;

        for optimizable_tensor in tensors {
            let theta = &optimizable_tensor.tensor();
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
            instructions.push(optimization_instruction!(
                OpCode::ScalarMul,
                OperatorAttributes::None,
                &[&beta1, &m],
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
                &[&beta2, &v],
                &[&tmp1],
            ));
            instructions.push(optimization_instruction!(
                OpCode::Mul,
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

            // m_hat = m / (1 - beta1**t)

            // m_multiplier = 1 / (1 - beta1**t)
            let m_multiplier = new_tensor!(device, 1, 1, vec![0.0])?;

            instructions.push(optimization_instruction!(
                OpCode::Pow,
                OperatorAttributes::None,
                &[&beta1, &t],
                &[&m_multiplier],
            ));
            instructions.push(optimization_instruction!(
                OpCode::Sub,
                OperatorAttributes::None,
                &[&one, &m_multiplier],
                &[&m_multiplier],
            ));
            instructions.push(optimization_instruction!(
                OpCode::Clip,
                OperatorAttributes::None,
                &[&epsilon, &f32_max, &m_multiplier],
                &[&m_multiplier],
            ));
            instructions.push(optimization_instruction!(
                OpCode::Div,
                OperatorAttributes::None,
                &[&one, &m_multiplier],
                &[&m_multiplier],
            ));

            let m_hat = new_tensor!(device, theta.rows(), theta.cols(), vec![0.0; theta.len()])?;

            // m_hatw
            instructions.push(optimization_instruction!(
                OpCode::ScalarMul,
                OperatorAttributes::None,
                &[&m_multiplier, &m],
                &[&m_hat],
            ));

            // v_hat = v / (1 - beta2**t)

            // v_multiplier = 1 / (1 - beta2**t)
            let v_multiplier = new_tensor!(device, 1, 1, vec![0.0])?;

            instructions.push(optimization_instruction!(
                OpCode::Pow,
                OperatorAttributes::None,
                &[&beta2, &t],
                &[&v_multiplier],
            ));
            instructions.push(optimization_instruction!(
                OpCode::Sub,
                OperatorAttributes::None,
                &[&one, &v_multiplier],
                &[&v_multiplier],
            ));
            instructions.push(optimization_instruction!(
                OpCode::Clip,
                OperatorAttributes::None,
                &[&epsilon, &f32_max, &v_multiplier],
                &[&v_multiplier],
            ));
            instructions.push(optimization_instruction!(
                OpCode::Div,
                OperatorAttributes::None,
                &[&one, &v_multiplier],
                &[&v_multiplier],
            ));

            let v_hat = new_tensor!(device, theta.rows(), theta.cols(), vec![0.0; theta.len()])?;

            // v_hat
            instructions.push(optimization_instruction!(
                OpCode::ScalarMul,
                OperatorAttributes::None,
                &[&v_multiplier, &v],
                &[&v_hat],
            ));

            // Update parameters with adaptive learning rate
            // theta = theta - alpha * m_hat / (sqrt(v_hat) + epsilon)

            instructions.push(optimization_instruction!(
                OpCode::Sqrt,
                OperatorAttributes::None,
                &[&tmp1],
                &[&tmp1],
            ));
            instructions.push(optimization_instruction!(
                OpCode::ScalarAdd,
                OperatorAttributes::None,
                &[&epsilon, &tmp1],
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
            // TODO try to remove this ClipNorm thing.
            instructions.push(optimization_instruction!(
                OpCode::ClipNorm,
                OperatorAttributes::None,
                &[&tmp1],
                &[&tmp1],
            ));

            instructions.push(optimization_instruction!(
                OpCode::ScalarMul,
                OperatorAttributes::None,
                &[&learning_rate, &tmp1],
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
