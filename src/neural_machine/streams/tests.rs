use crate::{
    mega_man_attention::MegaManAttentionModel, tensor::Error, Adam, BinaryOperator, Category,
    Device, NeuralMachine, OptimizerTrait, SoftmaxCrossEntropyLoss, Tokenizer, TokenizerTrait,
    UnaryModel,
};

use super::{make_simple_instructions, make_streams};

fn get_test_instructions() -> Result<Vec<(Vec<usize>, Vec<usize>)>, Error> {
    let device = Device::default();
    let tokenizer = Tokenizer::ascii_tokenizer();
    let vocab_size = tokenizer.vocab_size();
    let sequence_length = 32;
    let model = MegaManAttentionModel::new(&device, sequence_length, vocab_size)?;
    let model: Box<dyn UnaryModel> = Box::new(model);
    let loss_operator = SoftmaxCrossEntropyLoss::new(&device);
    let loss_operator: Box<dyn BinaryOperator> = Box::new(loss_operator);
    let clipped_gradient_norm = 1.0;
    let learning_rate = 0.05;
    let optimizer = Adam::new(learning_rate, 0.9, 0.98, 1e-9);
    let optimizer: Box<dyn OptimizerTrait> = Box::new(optimizer);
    let neural_machine = NeuralMachine::<f32>::try_new(
        &device,
        &model,
        &loss_operator,
        clipped_gradient_norm,
        &optimizer,
    )?;
    let instructions = vec![
        neural_machine.instructions(&Category::Inference),
        neural_machine.instructions(&Category::Loss),
        neural_machine.instructions(&Category::Gradient),
        neural_machine.instructions(&Category::Optimization),
    ]
    .concat();
    let simple_instructions = make_simple_instructions(&instructions);
    Ok(simple_instructions)
}

#[test]
fn each_instruction_is_executed_exactly_once() {
    let instructions = get_test_instructions().unwrap();
    let streams = make_streams(&instructions);
}

#[test]
fn each_operand_has_identical_read_and_write_access() {
    let instructions = get_test_instructions().unwrap();
    let streams = make_streams(&instructions);
}
