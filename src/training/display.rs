use crate::{
    get_row_argmaxes,
    perplexity::get_perplexity,
    tensor::{Error, Tensor},
    Tokenizer, TokenizerTrait,
};

pub trait TensorPrinter {
    fn print_expected_output_and_actual_output(
        &mut self,
        input: &Tensor,
        expected_output: &Tensor,
        actual_output: &Tensor,
    ) -> Result<(), Error>;
}

pub struct NextTokenPredictionPrinter {
    tokenizer: Tokenizer,
}

impl NextTokenPredictionPrinter {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self { tokenizer }
    }
}

impl TensorPrinter for NextTokenPredictionPrinter {
    fn print_expected_output_and_actual_output(
        &mut self,
        input: &Tensor,
        expected_output: &Tensor,
        actual_output: &Tensor,
    ) -> Result<(), Error> {
        let last_row = expected_output.rows() - 1;
        let input_tokens = get_row_argmaxes(input)?;
        let expected_output_argmaxes = get_row_argmaxes(expected_output)?;
        let expected_output_token = expected_output_argmaxes[last_row].to_owned();

        let actual_output_argmaxes = get_row_argmaxes(actual_output)?;
        let actual_output_token = actual_output_argmaxes[last_row].to_owned();

        let next_token_perplexity = get_perplexity(actual_output, actual_output.rows() - 1)?;

        let tokenizer: &mut Tokenizer = &mut self.tokenizer;
        println!(
            "  input_text: {}",
            tokens_to_text(&input_tokens, tokenizer)?
        );
        println!("  input_tokens: {:?}", &input_tokens);

        println!(
            "  expected_output_text: {}",
            tokens_to_text(&[expected_output_token], tokenizer)?
        );
        println!("  expected_output_token: {}", expected_output_token,);

        let actual_output_text: String = tokens_to_text(&[actual_output_token], tokenizer)?;
        println!(
            "  actual_output_text: {}",
            as_printable(actual_output_text, '?'),
        );
        println!("  actual_output_token: {}", actual_output_token);
        println!("  next_token_perplexity: {}", next_token_perplexity);

        let next_token_correctness = if expected_output_token == actual_output_token {
            1
        } else {
            0
        };
        println!("  next_token_correctness: {}", next_token_correctness);

        if expected_output.cols() < 10 {
            println!("expected_output {}", expected_output);
            println!("actual_output {}", actual_output);
        }

        Ok(())
    }
}

fn tokens_to_text(input_tokens: &[usize], tokenizer: &mut Tokenizer) -> Result<String, Error> {
    let input_text = tokenizer.decode(input_tokens)?;
    Ok(input_text)
}

trait IsPrintable {
    fn is_printable(&self) -> bool;
}

impl IsPrintable for char {
    fn is_printable(&self) -> bool {
        let code = *self as usize;
        if (32..=126).contains(&code) || code == 9 || code == 10 || code == 13 {
            return true;
        }
        false
    }
}

fn as_printable(output: String, replacement: char) -> String {
    let mut printable: String = String::new();
    for char in output.as_str().chars() {
        if char.is_printable() {
            printable += String::from(char).as_str();
        } else {
            printable += String::from(replacement).as_str();
        }
    }
    printable
}

pub struct RawPrinter {}

impl Default for RawPrinter {
    fn default() -> Self {
        Self {}
    }
}

impl TensorPrinter for RawPrinter {
    fn print_expected_output_and_actual_output(
        &mut self,
        input: &Tensor,
        expected_output: &Tensor,
        actual_output: &Tensor,
    ) -> Result<(), Error> {
        println!("input");
        println!("{}", input);

        println!("expected_output");
        println!("{}", expected_output);

        println!("actual_output");
        println!("{}", actual_output);

        Ok(())
    }
}

pub struct BoardPrinter {
    width: usize,
    height: usize,
}

impl BoardPrinter {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }
}

impl TensorPrinter for BoardPrinter {
    fn print_expected_output_and_actual_output(
        &mut self,
        input: &Tensor,
        expected_output: &Tensor,
        actual_output: &Tensor,
    ) -> Result<(), Error> {
        let input_tokens = get_row_argmaxes(input)?;
        let expected_output_tokens = get_row_argmaxes(expected_output)?;
        let actual_output_tokens = get_row_argmaxes(actual_output)?;

        let width = self.width;
        let height = self.height;

        println!("input");
        print_board(width, height, &input_tokens);

        println!("expected_output");
        print_board(width, height, &expected_output_tokens);

        println!("actual_output");
        print_board(width, height, &actual_output_tokens);

        let correctness = if expected_output_tokens == actual_output_tokens {
            1
        } else {
            0
        };
        println!("  correctness: {}", correctness);

        Ok(())
    }
}

fn print_board(width: usize, _height: usize, tokens: &[usize]) {
    for (i, token) in tokens.iter().enumerate() {
        if i % width == 0 && i != 0 {
            println!("");
        }
        print!(" {}", token);
    }
    println!("");
    println!("");
}
