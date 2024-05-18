use std::{io::Read, ops::Deref};

use rs_brain::{get_row_argmaxes, into_one_hot_encoded_rows, CrossEntropyLoss, Device, Embedding, Error, GradientDescent, Linear, LossOperator, Model, MultiHeadAttention, NeuralMachine, OptimizerTrait, Softmax, Tensor, TernaryOperator, Tokenizer, TokenizerTrait, UnaryModel, UnaryOperator};


struct ChatbotModel {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    vocab_size: usize,
    sequence_length: usize,
    embedding: Embedding,
    multi_head_attention: MultiHeadAttention,
    linear: Linear,
    softmax: Softmax,
}

impl UnaryModel for ChatbotModel {}

impl ChatbotModel {
    /// Attention Is All You Need
    /// https://arxiv.org/abs/1706.03762
    pub fn new(device: &Device) -> Self {
        let sequence_length = 32;
        let vocab_size = 256;
        let n_embd = 512;
        let num_heads = 8;

        let embedding = Embedding::new(device, vocab_size, n_embd);
        let multi_head_attention =
            MultiHeadAttention::try_new(device, sequence_length, n_embd, true, num_heads).unwrap();
        let linear = Linear::new(device, vocab_size, n_embd, true, sequence_length);
        let softmax = Softmax::new(device, true);

        Self {
            input_shape: vec![sequence_length, vocab_size],
            output_shape: vec![sequence_length, vocab_size],
            vocab_size,
            sequence_length,
            embedding,
            multi_head_attention,
            linear,
            softmax,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn sequence_length(&self) -> usize {
        self.sequence_length
    }
}

impl UnaryOperator for ChatbotModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
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
    let device = Device::cuda().unwrap();
    let model = ChatbotModel::new(&device);
    let vocab_size = model.vocab_size();
    let sequence_length = model.sequence_length();
    let model: Box<dyn UnaryModel> = Box::new(model);
    let clipped_gradient_norm = 1.0;
    let learning_rate = 0.05;
    let loss_operator: Box<dyn LossOperator> = Box::new(CrossEntropyLoss::new(&device));
    let optimizer: Box<dyn OptimizerTrait> = Box::new(GradientDescent::new(learning_rate));
    let chatbot = NeuralMachine::try_new(
        &device,
        &model,
        &loss_operator,
        clipped_gradient_norm,
        &optimizer,
    ).unwrap();

    let mut tokenizer = Tokenizer::ascii_tokenizer();

    let mut stdin_handle = std::io::stdin().lock();
    let mut prompt = String::new();

    while prompt.len() < sequence_length {
        let mut byte = [0u8];
        // Read a single byte
        stdin_handle.read_exact(&mut byte).unwrap();
    
        // Convert byte to char (may require additional logic for multi-byte characters)
        let character = byte[0] as char;
        prompt.push(character);
    }

    let prompt_tokens = tokenizer.encode(&prompt);
    let prompt_one_hot = into_one_hot_encoded_rows(&device, &prompt_tokens, vocab_size)?; 
    let chatbot_answer_one_hot = chatbot.infer(&prompt_one_hot)?;
    let chatbot_answer_tokens = get_row_argmaxes(&chatbot_answer_one_hot.tensor().deref().borrow())?;
    let chatbot_answer = tokenizer.decode(&chatbot_answer_tokens)?;
  
    println!("Chatbot: {}", chatbot_answer);
    Ok(())
}
