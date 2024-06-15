use novigrad::{
    error, get_row_argmax, into_one_hot_encoded_rows,
    neural_program::NeuralProgram,
    schedulers::DefaultStreamScheduler,
    tensor::{Error, ErrorEnum, Tensor},
    Adam, Device, Embedding, Linear, Model, MultiHeadAttention, NeuralMachine, Softmax,
    SoftmaxCrossEntropyLoss, TensorWithGrad, TernaryOperator, Tokenizer, TokenizerTrait,
    UnaryModel, UnaryOperator, WeightsInitialization,
};
use rand::prelude::SliceRandom;
use rand::thread_rng;
use std::{fs::read_to_string, io};

struct ChatbotModel {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    embedding: Embedding,
    multi_head_attention: MultiHeadAttention,
    linear: Linear,
    softmax: Softmax,
}

impl UnaryModel for ChatbotModel {}

impl ChatbotModel {
    /// Attention Is All You Need
    /// https://arxiv.org/abs/1706.03762
    pub fn new(device: &Device, sequence_length: usize, vocab_size: usize) -> Result<Self, Error> {
        let n_embd = 768;
        let num_heads = 12;
        let dropout_probability = 0.01;

        let embedding = Embedding::new(device, vocab_size, n_embd)?;
        let multi_head_attention = MultiHeadAttention::try_new(
            device,
            sequence_length,
            n_embd,
            true,
            num_heads,
            dropout_probability,
        )
        .unwrap();
        let linear = Linear::new(
            device,
            vocab_size,
            n_embd,
            WeightsInitialization::Kaiming,
            sequence_length,
        )?;
        let softmax = Softmax::new_with_next_is_cross_entropy_loss(device);

        let model = Self {
            input_shape: vec![sequence_length, vocab_size],
            output_shape: vec![sequence_length, vocab_size],
            embedding,
            multi_head_attention,
            linear,
            softmax,
        };
        Ok(model)
    }
}

impl UnaryOperator for ChatbotModel {
    fn forward(&self, input: &TensorWithGrad) -> Result<TensorWithGrad, Error> {
        let embedding = self.embedding.forward(input)?;
        let attentions = self
            .multi_head_attention
            .forward(&embedding, &embedding, &embedding)?;
        let linear = self.linear.forward(&attentions)?;
        let softmax = self.softmax.forward(&linear)?;
        Ok(softmax)
    }
}

impl Model for ChatbotModel {
    fn input_size(&self) -> Vec<usize> {
        self.input_shape.clone()
    }
    fn output_size(&self) -> Vec<usize> {
        self.output_shape.clone()
    }
}

fn main() -> Result<(), Error> {
    let device = Device::default();
    let mut tokenizer = Tokenizer::ascii_tokenizer();
    let sequence_length = 32;
    let vocab_size = tokenizer.vocab_size();
    let model = ChatbotModel::new(&device, sequence_length, vocab_size)?;
    let vocab_size = tokenizer.vocab_size();
    let loss_operator = SoftmaxCrossEntropyLoss::new(&device);
    let learning_rate = 0.05;
    let optimizer = Adam::new(learning_rate, 0.9, 0.98, 1e-9);
    let program = NeuralProgram::try_new(&device, &model, &loss_operator, &optimizer)?;
    let mut neural_machine =
        NeuralMachine::<f32, DefaultStreamScheduler>::try_new(&device, program).unwrap();

    println!("-------------------------------------------------------------------");
    println!("This is a Novigrad-powered chatbot");
    println!("A forward pass is all you need");
    println!("The chatbot knows nothing and will learn as you interact with it. (TODO)");
    println!("-------------------------------------------------------------------");

    let max_number_of_examples = 20;
    // From https://en.wikipedia.org/wiki/Geoffrey_Hinton
    let corpus = read_to_string("data/Geoffrey_Hinton.txt").unwrap()
        [0..(sequence_length + max_number_of_examples - 1)]
        .to_owned();

    println!("");
    println!("Corpus: {}", corpus);
    println!("");

    for turn in 0..100 {
        println!("Turn: {}", turn);

        // Learn things
        let end = if (sequence_length + 1) < corpus.len() {
            corpus.len() - (sequence_length + 1)
        } else {
            0
        };

        let mut indices = (0..end).collect::<Vec<_>>();
        indices.shuffle(&mut thread_rng());

        let mut total_loss = 0.0;
        for i in indices {
            let start = i;
            let end = start + sequence_length;

            let input = &corpus[start..end];
            let input_tokens = tokenizer.encode(&input);
            let input_one_hot = into_one_hot_encoded_rows(&device, &input_tokens, vocab_size)?;

            let expected_output = &corpus[start + 1..end + 1];
            let expected_output_tokens = tokenizer.encode(expected_output);
            let expected_output_one_hot =
                into_one_hot_encoded_rows(&device, &expected_output_tokens, vocab_size)?;

            let _actual_output_one_hot = neural_machine.infer(&input_one_hot)?;
            let loss = neural_machine.loss(&expected_output_one_hot)?;
            let loss: &Tensor = &loss.tensor();
            let loss: f32 = loss.try_into()?;
            total_loss += loss;
            neural_machine.compute_gradient()?;
            neural_machine.optimize()?;
        }
        println!("Loss: {}", total_loss);

        let start = 0;
        let prompt = &corpus[start..sequence_length];
        println!("Prompt: {}", prompt);
        let prompt_tokens = tokenizer.encode(&prompt);
        let max_len = corpus.len();
        let auto_regressive_tokens = auto_regressive_inference(
            &model,
            &mut neural_machine,
            &device,
            &prompt_tokens,
            max_len,
        )?;
        let actual_output = tokenizer.decode(&auto_regressive_tokens)?;

        println!("Chatbot: {}", actual_output);
    }

    Ok(())
}

fn _read_prompt() -> Result<String, Error> {
    let mut prompt = String::new();
    let stdin = io::stdin();
    match stdin.read_line(&mut prompt) {
        Ok(_) => Ok(prompt),
        Err(_) => Err(error!(ErrorEnum::InputOutputError)),
    }
}

fn auto_regressive_inference(
    model: &impl UnaryModel,
    neural_machine: &mut NeuralMachine<f32, DefaultStreamScheduler>,
    device: &Device,
    prompt_tokens: &[usize],
    max_len: usize,
) -> Result<Vec<usize>, Error> {
    let mut auto_regressive_tokens = vec![0 as usize; 0];
    for token in prompt_tokens {
        auto_regressive_tokens.push(token.clone());
    }
    let sequence_length = model.input_size()[0];
    let vocab_size = model.input_size()[1];
    // TODO implement another stopping criterion.
    while auto_regressive_tokens.len() < max_len {
        let input_tokens =
            &auto_regressive_tokens[(auto_regressive_tokens.len() - sequence_length)..];
        let input_one_hot = into_one_hot_encoded_rows(&device, input_tokens, vocab_size)?;

        let actual_output_one_hot = neural_machine.infer(&input_one_hot)?;
        let last_row = &actual_output_one_hot.tensor().rows() - 1;
        let predicted_next_token = get_row_argmax(&actual_output_one_hot.tensor(), last_row)?;
        auto_regressive_tokens.push(predicted_next_token);
    }
    Ok(auto_regressive_tokens)
}
