use novigrad::{
    datasets::into_one_hot_encoded_rows,
    error, get_row_argmax,
    neural_program::NeuralProgram,
    schedulers::DefaultStreamScheduler,
    tensor::{Error, ErrorEnum, Tensor},
    transformer_model::TransformerModel,
    Adam, Device, NeuralMachine, SoftmaxCrossEntropyLoss, TensorWithGrad, Tokenizer,
    TokenizerTrait, UnaryModel,
};
use rand::prelude::SliceRandom;
use rand::thread_rng;
use std::{fs::read_to_string, io};

fn main() -> Result<(), Error> {
    let device = Device::default();
    let mut tokenizer = Tokenizer::ascii_tokenizer();
    let context_length = 256;
    let _padding_token = 0;
    let layers = 2;
    let num_heads = 12;
    let dropout_probability = 0.1;
    let n_embd = 768;
    let vocab_size = tokenizer.vocab_size();
    let causal_mask = true;
    let model = TransformerModel::new(
        &device,
        layers,
        num_heads,
        dropout_probability,
        n_embd,
        context_length,
        vocab_size,
        causal_mask,
    )?;

    let loss_operator = SoftmaxCrossEntropyLoss::new(&device);
    let learning_rate = 0.05;
    let clipped_gradient_norm = true;
    let optimizer = Adam::new(learning_rate, 0.9, 0.98, 1e-9);
    let program = NeuralProgram::try_new(
        &device,
        &model,
        &loss_operator,
        &optimizer,
        clipped_gradient_norm,
    )?;

    let maximum_device_streams = 16;
    let mut neural_machine = NeuralMachine::<f32, DefaultStreamScheduler>::try_new(
        &device,
        program,
        maximum_device_streams,
    )
    .unwrap();

    println!("-------------------------------------------------------------------");
    println!("This is a Novigrad-powered chatbot");
    println!("A forward pass is all you need");
    println!("The chatbot knows nothing and will learn as you interact with it. (TODO)");
    println!("-------------------------------------------------------------------");

    //let train_corpus_path = "data/arc-prize-2024-3aa6fb7a-train-examples.txt";
    let train_corpus_path = "data/Geoffrey_Hinton.txt";
    let train_corpus = read_to_string(train_corpus_path).unwrap();

    println!();
    println!("Train Corpus: {}", train_corpus);
    println!();

    let train_examples = read_text_examples(&train_corpus);

    let train_examples = train_examples
        .iter()
        .map(|example| generate_examples(example, &mut tokenizer, context_length, &device))
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
        .concat();

    let mut indices = (0..train_examples.len()).collect::<Vec<_>>();
    for turn in 0..1000 {
        println!("Turn: {}", turn);

        indices.shuffle(&mut thread_rng());

        let mut total_loss = 0.0;
        println!("examples: {}", indices.len());
        for i in indices.iter() {
            let (input_one_hot, expected_output_one_hot) = &train_examples[*i];
            let _actual_output_one_hot = neural_machine.infer(&input_one_hot)?;
            let loss = neural_machine.loss(&expected_output_one_hot)?;
            let loss: &Tensor = &loss.tensor();
            let loss: f32 = loss.try_into()?;
            println!("i {} loss {}", i, loss);
            total_loss += loss;
            neural_machine.compute_gradient()?;
            neural_machine.optimize()?;
        }
        println!("Loss: {}", total_loss);

        /*
        let prompt = &train_corpus[0..64];
        println!("Prompt:  {}", prompt);
        let prompt_tokens = tokenizer.encode(prompt);
        let max_len = train_corpus.len();
        let auto_regressive_tokens = auto_regressive_inference(
            &model,
            &mut neural_machine,
            &device,
            &prompt_tokens,
            max_len,
        )?;
        let actual_output = tokenizer.decode(&auto_regressive_tokens)?;

        println!("Chatbot: {}", actual_output);
         */
    }

    Ok(())
}

pub fn add_padding(tokens: &mut Vec<usize>, context_length: usize, padding_token: usize) {
    while tokens.len() < context_length {
        tokens.push(padding_token);
    }
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
    let mut auto_regressive_tokens = vec![0_usize; 0];
    for token in prompt_tokens {
        auto_regressive_tokens.push(*token);
    }
    let context_length = model.input_size()[0];
    let vocab_size = model.input_size()[1];
    // TODO implement another stopping criterion.
    while auto_regressive_tokens.len() < max_len {
        let input_tokens = if auto_regressive_tokens.len() <= context_length {
            vec![
                auto_regressive_tokens.clone(),
                vec![0; context_length - auto_regressive_tokens.len()],
            ]
            .concat()
        } else {
            auto_regressive_tokens[(auto_regressive_tokens.len() - context_length)..].to_owned()
        };

        let input_one_hot = into_one_hot_encoded_rows(device, &input_tokens, vocab_size)?;

        let actual_output_one_hot = neural_machine.infer(&input_one_hot)?;
        let last_row = if auto_regressive_tokens.len() <= context_length {
            auto_regressive_tokens.len() - 1
        } else {
            context_length - 1
        };
        let predicted_next_token = get_row_argmax(&actual_output_one_hot.tensor(), last_row)?;
        auto_regressive_tokens.push(predicted_next_token);
    }
    Ok(auto_regressive_tokens)
}

fn read_text_examples(corpus: &str) -> Vec<String> {
    let begin_marker = "[example]";
    let end_marker = "[/example]";
    let mut examples = vec![];
    let mut cursor: &str = &corpus;
    let mut skipped = 0;
    let mut example_begin = cursor.find(begin_marker);
    while let Some(begin) = example_begin {
        let example_end = cursor.find(end_marker);
        if let Some(end) = example_end {
            let begin = skipped + begin;
            let end = skipped + end + end_marker.len();

            let example = corpus[begin..end].to_owned();
            examples.push(example);

            skipped += end;
            if skipped < corpus.len() {
                cursor = &corpus[skipped..];
                example_begin = cursor.find(begin_marker);
            } else {
                example_begin = None;
            }
        }
    }
    examples
}

fn generate_examples(
    example: &str,
    tokenizer: &mut Tokenizer,
    context_length: usize,
    device: &Device,
) -> Result<Vec<(TensorWithGrad, TensorWithGrad)>, Error> {
    let vocab_size = tokenizer.vocab_size();
    let tokens = tokenizer.encode(example);
    let mut examples = vec![];
    for i in 0..(tokens.len() - context_length) {
        let input_tokens = &tokens[i..i + context_length];
        let input_one_hot = into_one_hot_encoded_rows(&device, &input_tokens, vocab_size)?;

        let output_tokens = &tokens[i + 1..i + context_length + 1];
        let output_one_hot = into_one_hot_encoded_rows(&device, &output_tokens, vocab_size)?;

        examples.push((input_one_hot, output_one_hot));
    }
    Ok(examples)
}
