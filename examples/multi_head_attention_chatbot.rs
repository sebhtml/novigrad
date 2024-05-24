use novigrad::{
    error, get_row_argmax, into_one_hot_encoded_rows, BinaryOperator, CrossEntropyLoss, Device,
    Embedding, Error, ErrorEnum, GradientDescent, Linear, Model, MultiHeadAttention, NeuralMachine,
    OptimizerTrait, Softmax, Tensor, TensorWithGrad, TernaryOperator, Tokenizer, TokenizerTrait,
    UnaryModel, UnaryOperator,
};
use rand::prelude::SliceRandom;
use rand::thread_rng;
use std::{
    io::{self},
    ops::Deref,
};

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
        let dropout_probability = 0.1;

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
        let linear = Linear::new(device, vocab_size, n_embd, true, sequence_length)?;
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
    let debug = true;
    let print_in_console = true;
    let device = Device::cuda()?;
    //let device = Device::cpu();
    let mut tokenizer = Tokenizer::ascii_tokenizer();
    let sequence_length = 32;
    let vocab_size = tokenizer.vocab_size();
    let model = ChatbotModel::new(&device, sequence_length, vocab_size)?;
    let vocab_size = tokenizer.vocab_size();
    let model: Box<dyn UnaryModel> = Box::new(model);
    let clipped_gradient_norm = 1.0;
    let learning_rate = 0.05;
    let loss_operator: Box<dyn BinaryOperator> = Box::new(CrossEntropyLoss::new(&device));
    let optimizer: Box<dyn OptimizerTrait> = Box::new(GradientDescent::new(learning_rate));
    let chatbot = NeuralMachine::<f32>::try_new(
        &device,
        &model,
        &loss_operator,
        clipped_gradient_norm,
        &optimizer,
    )
    .unwrap();

    if print_in_console {
        println!("-------------------------------------------------------------------");
        println!("This is a Novigrad-powered chatbot");
        println!("A forward pass is all you need");
        println!("The chatbot knows nothing and will learn as you interact with it.");
        println!("-------------------------------------------------------------------");
    }

    for turn in 0..100 {
        if print_in_console {
            println!("Turn: {}", turn);
        }

        let mut prompt: String = if debug {
            let prompt =
                "Taylor Alison Swift (born December 13, 1989) is an American singer-songwriter.";
            prompt.into()
        } else {
            read_prompt()?
        };
        while prompt.len() < sequence_length {
            prompt += " ";
        }
        // Learn things
        let end = if (sequence_length + 1) < prompt.len() {
            prompt.len() - (sequence_length + 1)
        } else {
            0
        };

        let mut indices = (0..end).collect::<Vec<_>>();
        indices.shuffle(&mut thread_rng());

        for i in indices {
            let start = i;
            let end = start + sequence_length;

            let input = &prompt[start..end];
            let input_tokens = tokenizer.encode(&input);
            let input_one_hot = into_one_hot_encoded_rows(&device, &input_tokens, vocab_size)?;

            let expected_output = &prompt[start + 1..end + 1];
            let expected_output_tokens = tokenizer.encode(expected_output);
            let expected_output_one_hot =
                into_one_hot_encoded_rows(&device, &expected_output_tokens, vocab_size)?;

            let _actual_output_one_hot = chatbot.infer(&input_one_hot)?;
            let loss = chatbot.loss(&expected_output_one_hot)?;
            let loss: &Tensor = &loss.tensor().deref().borrow();
            let _loss: f32 = loss.try_into()?;
            if print_in_console {
                //println!("Loss {}", loss);
            }

            chatbot.compute_gradient()?;
            chatbot.optimize()?;
        }

        let input = &prompt[0..sequence_length];
        println!("Prompt: {}", input);
        let prompt_tokens = tokenizer.encode(&input);

        let mut auto_regressive_tokens = vec![0 as usize; 0];
        for token in prompt_tokens {
            auto_regressive_tokens.push(token);
        }

        // TODO implement another stopping criterion.
        while auto_regressive_tokens.len() < prompt.len() {
            let input_tokens =
                &auto_regressive_tokens[(auto_regressive_tokens.len() - sequence_length)..];
            let input_one_hot = into_one_hot_encoded_rows(&device, input_tokens, vocab_size)?;

            let actual_output_one_hot = chatbot.infer(&input_one_hot)?;
            let last_row = &actual_output_one_hot.tensor().deref().borrow().rows() - 1;
            let predicted_next_token =
                get_row_argmax(&actual_output_one_hot.tensor().deref().borrow(), last_row)?;
            auto_regressive_tokens.push(predicted_next_token);
        }

        let actual_output = tokenizer.decode(&auto_regressive_tokens)?;

        println!("Chatbot: {}", actual_output);
    }

    Ok(())
}

fn read_prompt() -> Result<String, Error> {
    let mut prompt = String::new();
    let stdin = io::stdin();
    match stdin.read_line(&mut prompt) {
        Ok(_) => Ok(prompt),
        Err(_) => Err(error!(ErrorEnum::InputOutputError)),
    }
}
