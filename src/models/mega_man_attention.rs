use super::load_examples;
use crate::{
    CrossEntropyLoss, Device, NeuralMachine, TernaryOperator, Tokenizer, TokenizerTrait,
    UnaryOperator,
};
use crate::{DatasetDetails, Error};

use std::ops::Deref;

use crate::{Embedding, Linear, Model, ScaledDotProductAttention, Softmax, Tensor};

struct MegaManAttentionModel {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    vocab_size: usize,
    sequence_length: usize,
    embedding: Embedding,
    q: Linear,
    k: Linear,
    v: Linear,
    attention: ScaledDotProductAttention,
    linear: Linear,
    softmax: Softmax,
}

impl MegaManAttentionModel {
    pub fn new(device: &Device) -> Self {
        let _batch_size = 1;
        let sequence_length = 6;
        let vocab_size = 20;
        let n_embd = 4;
        let _num_heads = 1;
        let _n_layer = 1;
        let _dropout = 0.1;
        let _block_size = 2048;

        let q = Linear::new(device, n_embd, n_embd, sequence_length);
        let k = Linear::new(device, n_embd, n_embd, sequence_length);
        let v = Linear::new(device, n_embd, n_embd, sequence_length);

        let attention =
            ScaledDotProductAttention::try_new(device, sequence_length, n_embd, true).unwrap();

        let linear = Linear::new(device, vocab_size, n_embd, sequence_length);

        Self {
            input_shape: vec![sequence_length, vocab_size],
            output_shape: vec![sequence_length, vocab_size],
            vocab_size,
            sequence_length,
            embedding: Embedding::new(device, vocab_size, n_embd),
            q,
            k,
            v,
            attention,
            linear,
            softmax: Softmax::new(device, true),
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn sequence_length(&self) -> usize {
        self.sequence_length
    }
}

impl UnaryOperator for MegaManAttentionModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        let debug = false;
        if debug {
            println!("----");
        }
        if debug {
            println!("input {}", input.tensor().deref().borrow());
        }
        let embeddings = self.embedding.forward(input)?;
        if debug {
            println!("embedding {}", &embeddings.tensor().deref().borrow());
        }
        let q = self.q.forward(&embeddings)?;
        if debug {
            println!("q {}", &q.tensor().deref().borrow());
        }
        let k = self.k.forward(&embeddings)?;
        if debug {
            println!("k {}", &k.tensor().deref().borrow());
        }
        let v = self.v.forward(&embeddings)?;
        if debug {
            println!("v {}", &v.tensor().deref().borrow());
        }
        let attended = self.attention.forward(&q, &k, &v)?;
        if debug {
            println!("attended {}", &attended.tensor().deref().borrow());
        }
        let linearized = self.linear.forward(&attended)?;
        if debug {
            println!("linearized {}", &linearized.tensor().deref().borrow());
        }
        let probabilities = self.softmax.forward(&linearized)?;
        if debug {
            println!("probabilities {}", &probabilities.tensor().deref().borrow());
        }
        Ok(probabilities)
    }
}

impl Model for MegaManAttentionModel {
    fn input_size(&self) -> Vec<usize> {
        self.input_shape.clone()
    }

    fn output_size(&self) -> Vec<usize> {
        self.output_shape.clone()
    }
}

pub fn load_dataset(device: &Device) -> Result<DatasetDetails, Error> {
    let file_path = "data/Mega_Man.txt";
    let max_chars = Some(30);
    let max_number_of_examples = 10;
    // TODO vocab_size should be a new argument
    let model = MegaManAttentionModel::new(device);
    let vocab_size = model.vocab_size();
    let mut tokenizer = Tokenizer::byte_pair_encoding();

    let input_sequence_length = model.sequence_length();
    let output_sequence_length = input_sequence_length;
    let examples = load_examples(
        &device,
        file_path,
        max_chars,
        max_number_of_examples,
        input_sequence_length,
        output_sequence_length,
        vocab_size,
        &mut tokenizer,
    )?;

    println!("TOkenizer vocab_size: {}", tokenizer.vocab_size());

    let model = model;
    let loss_operator = CrossEntropyLoss::new(device);
    let program = NeuralMachine::try_new(&device, &model, &loss_operator)?;

    let details = DatasetDetails {
        device: device.clone(),
        tokenizer,
        examples,
        program,
        epochs: 1000,
        progress: 100,
        initial_total_error_min: 50.0,
        final_total_error_max: 0.002,
        learning_rate: 0.5,
    };
    Ok(details)
}
