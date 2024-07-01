use crate::{
    display::NextTokenPredictionPrinter, tensor::Error, transformer_model::TransformerModel,
    Device, GradientDescent, Metrics, SoftmaxCrossEntropyLoss, Tokenizer, TokenizerTrait,
};

use super::{load_examples, DatasetDetails};

pub fn load_geoffroy_hinton_dataset(
    device: &Device,
) -> Result<
    DatasetDetails<
        TransformerModel,
        SoftmaxCrossEntropyLoss,
        GradientDescent,
        NextTokenPredictionPrinter,
    >,
    Error,
> {
    let file_path = "data/Geoffrey_Hinton.txt";
    let max_chars = None;
    let max_number_of_examples = 30;
    let mut tokenizer = Tokenizer::ascii_tokenizer();
    let context_length = 64;

    let input_sequence_length = context_length;
    let output_sequence_length = context_length;
    let examples = load_examples(
        device,
        file_path,
        max_chars,
        max_number_of_examples,
        input_sequence_length,
        output_sequence_length,
        &mut tokenizer,
    )?;

    let vocab_size = tokenizer.vocab_size();
    let layers = 1;
    let causal_mask = true;
    let num_heads = 12;
    let dropout_probability = 0.1;
    let n_embd = 768;
    let model = TransformerModel::new(
        device,
        layers,
        num_heads,
        dropout_probability,
        n_embd,
        context_length,
        vocab_size,
        causal_mask,
    )?;

    let loss_operator = SoftmaxCrossEntropyLoss::new(device);
    let learning_rate = 0.2;
    //let optimizer = Adam::new(learning_rate, 0.9, 0.98, 1e-9);
    let optimizer = GradientDescent::new(learning_rate);
    let details = DatasetDetails {
        device: device.clone(),
        train_examples: examples,
        test_examples: vec![],
        model,
        loss_operator,
        optimizer,
        epochs: 200,
        progress: 10,
        learning_rate,
        shuffle_examples: true,
        clip_gradient_norm: true,
        initial_metrics_min: Metrics {
            total_loss: 4000.0,
            total_next_token_perplexity: 5.0,
        },
        final_metrics_max: Metrics {
            total_loss: 350.0,
            total_next_token_perplexity: 20.0,
        },
        maximum_incorrect_predicted_next_tokens: 0,
        printer: NextTokenPredictionPrinter::new(tokenizer),
        batch_size: 1,
    };
    Ok(details)
}
