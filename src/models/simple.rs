use crate::{
    error, into_one_hot_encoded_rows, tensor::Error, tensor::ErrorEnum, Device, GradientDescent,
    Metrics, ModelDetails, SoftmaxCrossEntropyLoss, TensorWithGrad, Tokenizer, TokenizerTrait,
    UnaryModel, UnaryOperator,
};
use crate::{Embedding, Linear, Model, Reshape, Sigmoid, Softmax, WeightsInitialization};

pub struct SimpleModel {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    embedding: Embedding,
    linear_0: Linear,
    sigmoid_0: Sigmoid,
    reshape: Reshape,
    linear_1: Linear,
    sigmoid_1: Sigmoid,
    linear_2: Linear,
    softmax: Softmax,
}

impl UnaryModel for SimpleModel {}

impl SimpleModel {
    pub fn new(device: &Device, sequence_length: usize, vocab_size: usize) -> Result<Self, Error> {
        let n_embd = 384;
        let output_rows = 1;

        let embedding = Embedding::new(device, vocab_size, n_embd)?;
        let linear_0 = Linear::new(
            device,
            n_embd,
            n_embd,
            WeightsInitialization::Kaiming,
            sequence_length,
        )?;
        let sigmoid_0 = Sigmoid::new(device);
        let reshape = Reshape::new(
            device,
            vec![sequence_length, n_embd],
            vec![output_rows, sequence_length * n_embd],
        );
        let linear_1 = Linear::new(
            device,
            n_embd,
            sequence_length * n_embd,
            WeightsInitialization::Kaiming,
            output_rows,
        )?;
        let sigmoid_1 = Sigmoid::new(device);
        let linear_2 = Linear::new(
            device,
            vocab_size,
            n_embd,
            WeightsInitialization::Kaiming,
            output_rows,
        )?;
        let softmax = Softmax::new_with_next_is_cross_entropy_loss(device);

        let model = Self {
            input_shape: vec![sequence_length, vocab_size],
            output_shape: vec![output_rows, vocab_size],
            embedding,
            linear_0,
            sigmoid_0,
            reshape,
            linear_1,
            sigmoid_1,
            linear_2,
            softmax,
        };
        Ok(model)
    }
}

impl UnaryOperator for SimpleModel {
    fn forward(&self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
        let state_0 = self.embedding.forward(input)?;
        let state_1 = self.linear_0.forward(&state_0)?;
        let state_2 = self.sigmoid_0.forward(&state_1)?;
        let state_3 = self.reshape.forward(&state_2)?;
        let state_4 = self.linear_1.forward(&state_3)?;
        let state_5 = self.sigmoid_1.forward(&state_4)?;
        let state_6 = self.linear_2.forward(&state_5)?;
        let state_7 = self.softmax.forward(&state_6)?;
        Ok(state_7)
    }
}

impl Model for SimpleModel {
    fn input_size(&self) -> Vec<usize> {
        self.input_shape.clone()
    }
    fn output_size(&self) -> Vec<usize> {
        self.output_shape.clone()
    }
}

fn load_examples(
    device: &Device,
    tokenizer: &mut Tokenizer,
) -> Result<Vec<(TensorWithGrad, TensorWithGrad)>, Error> {
    let examples: Vec<_> = ["quizzed", "fuzzing"]
        .iter()
        .map(|text| {
            (
                tokenizer.encode(&text[0..text.len() - 1]),
                tokenizer.encode(&text[text.len() - 1..text.len()]),
            )
        })
        .collect();

    let vocab_size = tokenizer.vocab_size();
    

    examples
        .into_iter()
        .map(|example| {
            let one_hot_encoded_input = into_one_hot_encoded_rows(device, &example.0, vocab_size);
            let one_hot_encoded_output = into_one_hot_encoded_rows(device, &example.1, vocab_size);
            (one_hot_encoded_input, one_hot_encoded_output)
        })
        .try_fold(vec![], |mut acc, item| match item {
            (Ok(a), Ok(b)) => {
                acc.push((a, b));
                Ok(acc)
            }
            _ => Err(error!(ErrorEnum::UnsupportedOperation)),
        })
}

pub fn load_simple_model(
    device: &Device,
) -> Result<ModelDetails<SimpleModel, SoftmaxCrossEntropyLoss, GradientDescent>, Error> {
    let mut tokenizer = Tokenizer::ascii_tokenizer();
    let sequence_length = 6;
    let examples = load_examples(device, &mut tokenizer)?;

    let loss_operator = SoftmaxCrossEntropyLoss::new(device);
    let learning_rate = 0.5;
    let vocab_size = tokenizer.vocab_size();
    let model = SimpleModel::new(device, sequence_length, vocab_size)?;
    let optimizer = GradientDescent::new(learning_rate);
    let details = ModelDetails {
        device: device.clone(),
        tokenizer: Some(tokenizer),
        examples,
        model,
        loss_operator,
        optimizer,
        epochs: 500,
        progress: 100,
        learning_rate,
        shuffle_examples: true,
        clipped_gradient_norm: 1.0,
        initial_metrics: Metrics {
            total_loss: 5.0,
            total_perplexity: 200.0,
        },
        final_metrics: Metrics {
            total_loss: 0.0,
            total_perplexity: 2.0,
        },
    };
    Ok(details)
}
