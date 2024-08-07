use novigrad::{
    batch::make_batches,
    datasets::into_one_hot_encoded_rows,
    error, get_row_argmax,
    neural_program::NeuralProgram,
    schedulers::DefaultStreamScheduler,
    tensor::{Error, ErrorEnum, Tensor},
    transformer_model::TransformerModel,
    Adam, Device, NeuralMachine, SoftmaxCrossEntropyLoss, TensorWithGrad, Tokenizer,
    TokenizerTrait,
};
use std::{fs::read_to_string, io};

fn main() -> Result<(), Error> {
    let device = Device::default();
    let mut tokenizer = Tokenizer::ascii_tokenizer();
    let sequence_length = 32; //256;
    let padding_token = 0;
    let layers = 1;
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
        sequence_length,
        vocab_size,
        causal_mask,
    )?;

    let loss_operator = SoftmaxCrossEntropyLoss::new(&device);
    let batch_size = 32;
    let clip_grad_norm = true;
    let optimizer = Adam::try_new(0.2, 0.9, 0.999, 1e-8, 0.0)?;
    let program = NeuralProgram::try_new(
        &device,
        &model,
        &loss_operator,
        &optimizer,
        clip_grad_norm,
        batch_size,
    )?;

    let maximum_device_streams = 16;
    let epochs = 100;
    let shuffle_examples = true;
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

    let train_corpus_path = "data/Geoffrey_Hinton.txt";
    let train_corpus = read_to_string(train_corpus_path).unwrap();

    println!();
    println!("Train Corpus: {}", train_corpus);
    println!();

    let train_examples = read_text_examples(&train_corpus);

    let train_examples = train_examples
        .iter()
        .map(|example| generate_examples(example, &mut tokenizer, sequence_length, &device))
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
        .concat();

    let indices = (0..train_examples.len()).collect::<Vec<_>>();

    for epoch in 0..epochs {
        println!("Epoch: {} / {}", epoch, epochs);

        let batches = make_batches(&indices, shuffle_examples, batch_size);
        let mut total_loss = 0.0;

        for batch in batches.iter() {
            for i in batch.iter() {
                let (input_one_hot, expected_output_one_hot) = &train_examples[*i];
                let _actual_output_one_hot = neural_machine.infer(&input_one_hot)?;
                let loss = neural_machine.loss(&expected_output_one_hot)?;
                let loss: &Tensor = &loss.tensor();
                let loss: f32 = loss.try_into()?;

                total_loss += loss;
                neural_machine.compute_gradient()?;
            }
            neural_machine.optimize()?;
        }
        println!("Loss: {}", total_loss);

        let prompt = &train_corpus[0..25];
        println!("Prompt:  {}", prompt);
        let prompt_tokens = tokenizer.encode(prompt);
        let max_len = 60;
        let auto_regressive_tokens = auto_regressive_inference(
            &mut neural_machine,
            &device,
            &prompt_tokens,
            sequence_length,
            vocab_size,
            max_len,
            padding_token,
        )?;
        let actual_output = tokenizer.decode(&auto_regressive_tokens)?;

        println!("Chatbot: {}", actual_output);
    }

    Ok(())
}

pub fn add_padding(tokens: &mut Vec<usize>, sequence_length: usize, padding_token: usize) {
    while tokens.len() < sequence_length {
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
    neural_machine: &mut NeuralMachine<f32, DefaultStreamScheduler>,
    device: &Device,
    prompt_tokens: &[usize],
    sequence_length: usize,
    vocab_size: usize,
    max_len: usize,
    padding_token: usize,
) -> Result<Vec<usize>, Error> {
    let mut auto_regressive_tokens = prompt_tokens.to_owned();

    // TODO implement another stopping criterion.
    while auto_regressive_tokens.len() < max_len {
        let input_tokens = if auto_regressive_tokens.len() <= sequence_length {
            vec![
                auto_regressive_tokens.clone(),
                vec![padding_token; sequence_length - auto_regressive_tokens.len()],
            ]
            .concat()
        } else {
            auto_regressive_tokens[(auto_regressive_tokens.len() - sequence_length)..].to_owned()
        };

        let input_one_hot = into_one_hot_encoded_rows(device, &input_tokens, vocab_size)?;

        let actual_output_one_hot = neural_machine.infer(&input_one_hot)?;
        let last_row = if auto_regressive_tokens.len() <= sequence_length {
            auto_regressive_tokens.len() - 1
        } else {
            sequence_length - 1
        };
        let predicted_next_token = get_row_argmax(&actual_output_one_hot.tensor(), last_row)?;
        //println!("predicted next token: {}", predicted_next_token);
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
    sequence_length: usize,
    device: &Device,
) -> Result<Vec<(TensorWithGrad, TensorWithGrad)>, Error> {
    let vocab_size = tokenizer.vocab_size();
    let tokens = tokenizer.encode(example);
    let mut examples = vec![];
    for i in 0..(tokens.len() - sequence_length) {
        let input_tokens = &tokens[i..i + sequence_length];
        let input_one_hot = into_one_hot_encoded_rows(&device, &input_tokens, vocab_size)?;

        let output_tokens = &tokens[i + 1..i + sequence_length + 1];
        let output_one_hot = into_one_hot_encoded_rows(&device, &output_tokens, vocab_size)?;

        //println!("in {:?}", input_tokens);
        //println!("out {:?}", output_tokens);
        examples.push((input_one_hot, output_one_hot));
    }
    Ok(examples)
}
