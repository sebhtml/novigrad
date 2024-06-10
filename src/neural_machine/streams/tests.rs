use std::{ops::Deref, sync::Arc};

use crate::{
    mega_man_attention::MegaManAttentionModel,
    neural_machine::streams::{
        instruction::print_instructions,
        stream::{make_streams, print_streams},
    },
    neural_program::NeuralProgram,
    tensor::Error,
    Adam, Category, Device, Instruction, SoftmaxCrossEntropyLoss, Tokenizer, TokenizerTrait,
};
use test_case::test_case;

use super::{
    instruction::make_simple_instructions,
    scheduler::simulate_execution_and_collect_transactions,
    transaction::{get_all_instruction_transactions, get_operand_transaction_pairs, Access},
};

fn get_test_instructions(filter: Option<Category>) -> Result<Vec<Instruction>, Error> {
    let device = Device::default();
    let tokenizer = Tokenizer::ascii_tokenizer();
    let vocab_size = tokenizer.vocab_size();
    let sequence_length = 32;
    let model = MegaManAttentionModel::new(&device, sequence_length, vocab_size)?;
    let loss_operator = SoftmaxCrossEntropyLoss::new(&device);
    let learning_rate = 0.05;
    let optimizer = Adam::new(learning_rate, 0.9, 0.98, 1e-9);
    let program = NeuralProgram::try_new(&device, &model, &loss_operator, &optimizer)?;
    let instructions = program.instructions;
    let instructions = match filter {
        Some(category) => instructions
            .into_iter()
            .filter(|x| x.category() == category)
            .collect(),
        None => instructions,
    };
    Ok(instructions)
}

#[test_case(None ; "no category filter")]
#[test_case(Some(Category::Inference) ; "inference filter")]
#[test_case(Some(Category::Loss) ; "loss filter")]
#[test_case(Some(Category::Gradient) ; "gradient filter")]
#[test_case(Some(Category::Optimization) ; "optimization filter")]
fn each_instruction_is_executed_exactly_once(filter: Option<Category>) {
    let instructions = get_test_instructions(filter).unwrap();
    let simple_instructions = make_simple_instructions(&instructions);
    let expected_instructions = (0..instructions.len()).collect::<Vec<_>>();
    let minimum_write_before_read_for_new_stream = 4;
    let minimum_dependents_for_stream = 12;
    let minimum_stream_instructions = 32;
    let streams = make_streams(
        &simple_instructions,
        minimum_write_before_read_for_new_stream,
        minimum_dependents_for_stream,
        minimum_stream_instructions,
    );
    let mut actual_instructions = streams
        .iter()
        .map(|x| x.instructions.deref().clone())
        .collect::<Vec<_>>()
        .concat();
    actual_instructions.sort();
    assert_eq!(expected_instructions, actual_instructions);
}

#[test]
fn the_instructions_length_is_correct() {
    let instructions = get_test_instructions(None).unwrap();
    let simple_instructions = make_simple_instructions(&instructions);
    assert_eq!(2635, instructions.len());
    let minimum_write_before_read_for_new_stream = 4;
    let minimum_dependents_for_stream = 12;
    let minimum_stream_instructions = 32;
    let streams = make_streams(
        &simple_instructions,
        minimum_write_before_read_for_new_stream,
        minimum_dependents_for_stream,
        minimum_stream_instructions,
    );
    let actual_instructions = streams
        .iter()
        .map(|i| i.instructions.deref().clone())
        .collect::<Vec<Vec<usize>>>()
        .concat();
    assert_eq!(2635, actual_instructions.len());
}

#[test]
fn the_streams_length_are_correct() {
    let instructions = get_test_instructions(None).unwrap();
    let simple_instructions = make_simple_instructions(&instructions);
    let minimum_write_before_read_for_new_stream = 4;
    let minimum_dependents_for_stream = 12;
    let minimum_stream_instructions = 32;
    let streams = make_streams(
        &simple_instructions,
        minimum_write_before_read_for_new_stream,
        minimum_dependents_for_stream,
        minimum_stream_instructions,
    );
    print_streams("test", &streams);
    assert_eq!(
        107,
        streams.iter().filter(|x| x.instructions.len() > 0).count()
    );
}

#[test]
fn simple_problem_for_streams() {
    let instructions = vec![
        (vec![0, 1], vec![2]),
        (vec![3, 4], vec![5]),
        (vec![6, 7], vec![8]),
        (vec![9, 10], vec![11]),
        (vec![2, 5, 8, 11], vec![12]),
        (vec![12, 13], vec![14]),
    ];
    let minimum_write_before_read_for_new_stream = 4;
    let minimum_dependents_for_stream = 12;
    let minimum_stream_instructions = 32;
    let streams = make_streams(
        &instructions,
        minimum_write_before_read_for_new_stream,
        minimum_dependents_for_stream,
        minimum_stream_instructions,
    );

    print_instructions(&instructions);
    print_streams("test", &streams);

    assert_eq!(5, streams.len());

    assert_eq!(vec![] as Vec<usize>, *streams[0].dependencies);
    assert_eq!(vec![] as Vec<usize>, *streams[1].dependencies);
    assert_eq!(vec![] as Vec<usize>, *streams[2].dependencies);
    assert_eq!(vec![] as Vec<usize>, *streams[3].dependencies);
    assert_eq!(vec![0, 1, 2, 3], *streams[4].dependencies);

    assert_eq!(vec![0], *streams[0].instructions);
    assert_eq!(vec![1], *streams[1].instructions);
    assert_eq!(vec![2], *streams[2].instructions);
    assert_eq!(vec![3], *streams[3].instructions);
    assert_eq!(vec![4, 5], *streams[4].instructions);
}

#[test]
fn problem_2_for_streams_with_fuse() {
    let instructions = vec![
        (vec![0], vec![1]),
        (vec![1], vec![2]),
        (vec![1], vec![3]),
        (vec![1], vec![4]),
        (vec![1], vec![5]),
        (vec![2, 3, 4, 5], vec![6]),
        (vec![6], vec![7]),
    ];

    let minimum_write_before_read_for_new_stream = 4;
    let minimum_dependents_for_stream = 12;
    let minimum_stream_instructions = 2;
    let streams = make_streams(
        &instructions,
        minimum_write_before_read_for_new_stream,
        minimum_dependents_for_stream,
        minimum_stream_instructions,
    );

    print_instructions(&instructions);
    print_streams("streams", &streams);

    assert_eq!(6, streams.len());

    assert_eq!(vec![4], *streams[0].dependencies);
    assert_eq!(vec![4], *streams[1].dependencies);
    assert_eq!(vec![4], *streams[2].dependencies);
    assert_eq!(vec![4], *streams[3].dependencies);
    assert_eq!(vec![] as Vec<usize>, *streams[4].dependencies);
    assert_eq!(vec![0, 1, 2, 3], *streams[5].dependencies);

    assert_eq!(vec![1], *streams[0].instructions);
    assert_eq!(vec![2], *streams[1].instructions);
    assert_eq!(vec![3], *streams[2].instructions);
    assert_eq!(vec![4], *streams[3].instructions);
    assert_eq!(vec![0], *streams[4].instructions);
    assert_eq!(vec![5, 6], *streams[5].instructions);
}

#[test]
fn problem_2_for_streams_no_fuse() {
    let instructions = vec![
        (vec![0], vec![1]),
        (vec![1], vec![2]),
        (vec![1], vec![3]),
        (vec![1], vec![4]),
        (vec![1], vec![5]),
        (vec![2, 3, 4, 5], vec![6]),
        (vec![6], vec![7]),
    ];

    let minimum_write_before_read_for_new_stream = 4;
    let minimum_dependents_for_stream = 12;
    let minimum_stream_instructions = 1;
    let streams = make_streams(
        &instructions,
        minimum_write_before_read_for_new_stream,
        minimum_dependents_for_stream,
        minimum_stream_instructions,
    );

    print_instructions(&instructions);
    print_streams("streams", &streams);

    assert_eq!(7, streams.len());

    assert_eq!(vec![4], *streams[0].dependencies);
    assert_eq!(vec![4], *streams[1].dependencies);
    assert_eq!(vec![4], *streams[2].dependencies);
    assert_eq!(vec![4], *streams[3].dependencies);
    assert_eq!(vec![] as Vec<usize>, *streams[4].dependencies);
    assert_eq!(vec![0, 1, 2, 3], *streams[5].dependencies);

    assert_eq!(vec![1], *streams[0].instructions);
    assert_eq!(vec![2], *streams[1].instructions);
    assert_eq!(vec![3], *streams[2].instructions);
    assert_eq!(vec![4], *streams[3].instructions);
    assert_eq!(vec![0], *streams[4].instructions);
    assert_eq!(vec![5], *streams[5].instructions);
    assert_eq!(vec![6], *streams[6].instructions);
}

#[test]
fn problem_3_for_streams() {
    let instructions = vec![
        (vec![0], vec![1]),
        (vec![1], vec![2]),
        (vec![1], vec![3]),
        (vec![1], vec![4]),
        (vec![1], vec![5]),
        (vec![2, 3, 4, 5], vec![6]),
        (vec![6], vec![7]),
    ];
    let minimum_write_before_read_for_new_stream = 4;
    let minimum_dependents_for_stream = 12;
    let minimum_stream_instructions = 32;
    let streams = make_streams(
        &instructions,
        minimum_write_before_read_for_new_stream,
        minimum_dependents_for_stream,
        minimum_stream_instructions,
    );

    print_streams("test", &streams);

    assert_eq!(6, streams.len());

    assert_eq!(vec![4], *streams[0].dependencies);
    assert_eq!(vec![4], *streams[1].dependencies);
    assert_eq!(vec![4], *streams[2].dependencies);
    assert_eq!(vec![4], *streams[3].dependencies);
    assert_eq!(vec![] as Vec::<usize>, *streams[4].dependencies);
    assert_eq!(vec![0, 1, 2, 3], *streams[5].dependencies);

    assert_eq!(vec![1], *streams[0].instructions);
    assert_eq!(vec![2], *streams[1].instructions);
    assert_eq!(vec![3], *streams[2].instructions);
    assert_eq!(vec![4], *streams[3].instructions);
    assert_eq!(vec![0], *streams[4].instructions);
    assert_eq!(vec![5, 6], *streams[5].instructions);
}

#[test]
fn problem_4_for_streams() {
    let instructions = vec![
        (vec![0], vec![1]),
        (vec![1], vec![2]),
        (vec![1], vec![3]),
        (vec![1], vec![4]),
        (vec![1], vec![5]),
        (vec![2, 3, 4, 5], vec![6]),
        (vec![6], vec![7]),
        (vec![7], vec![8]),
        (vec![7], vec![9]),
        (vec![7], vec![10]),
        (vec![7], vec![11]),
        (vec![8, 9, 10, 11], vec![12]),
    ];
    let minimum_write_before_read_for_new_stream = 4;
    let minimum_dependents_for_stream = 12;
    let minimum_stream_instructions = 32;
    let streams = make_streams(
        &instructions,
        minimum_write_before_read_for_new_stream,
        minimum_dependents_for_stream,
        minimum_stream_instructions,
    );

    print_streams("test", &streams);

    assert_eq!(11, streams.len());

    assert_eq!(vec![8], *streams[0].dependencies);
    assert_eq!(vec![8], *streams[1].dependencies);
    assert_eq!(vec![8], *streams[2].dependencies);
    assert_eq!(vec![8], *streams[3].dependencies);

    assert_eq!(vec![9], *streams[4].dependencies);
    assert_eq!(vec![9], *streams[5].dependencies);
    assert_eq!(vec![9], *streams[6].dependencies);
    assert_eq!(vec![9], *streams[7].dependencies);

    assert_eq!(vec![] as Vec::<usize>, *streams[8].dependencies);

    assert_eq!(vec![0, 1, 2, 3], *streams[9].dependencies);
    assert_eq!(vec![4, 5, 6, 7], *streams[10].dependencies);

    assert_eq!(vec![1], *streams[0].instructions);
    assert_eq!(vec![2], *streams[1].instructions);
    assert_eq!(vec![3], *streams[2].instructions);
    assert_eq!(vec![4], *streams[3].instructions);

    assert_eq!(vec![7], *streams[4].instructions);
    assert_eq!(vec![8], *streams[5].instructions);
    assert_eq!(vec![9], *streams[6].instructions);
    assert_eq!(vec![10], *streams[7].instructions);

    assert_eq!(vec![0], *streams[8].instructions);
    assert_eq!(vec![5, 6], *streams[9].instructions);
    assert_eq!(vec![11], *streams[10].instructions);
}

#[test]
fn many_independent_instructions_in_two_streams() {
    let instructions = vec![(vec![0, 1], vec![2]), (vec![3, 4], vec![5])];
    let minimum_write_before_read_for_new_stream = 4;
    let minimum_dependents_for_stream = 12;
    let minimum_stream_instructions = 1;
    let streams = make_streams(
        &instructions,
        minimum_write_before_read_for_new_stream,
        minimum_dependents_for_stream,
        minimum_stream_instructions,
    );

    print_streams("test", &streams);

    assert_eq!(2, streams.len());

    assert_eq!(vec![0; 0], *streams[0].dependencies);
    assert_eq!(vec![0; 0], *streams[0].dependencies);

    assert_eq!(vec![0], *streams[0].instructions);
    assert_eq!(vec![1], *streams[1].instructions);
}

#[test]
fn many_independent_instructions_in_one_stream() {
    let instructions = vec![
        (vec![0, 1], vec![2]), //
        (vec![3, 4], vec![5]), //
    ];
    let minimum_write_before_read_for_new_stream = 4;
    let minimum_dependents_for_stream = 12;
    let minimum_stream_instructions = 32;
    let streams = make_streams(
        &instructions,
        minimum_write_before_read_for_new_stream,
        minimum_dependents_for_stream,
        minimum_stream_instructions,
    );

    print_streams("test", &streams);

    assert_eq!(1, streams.len());

    assert_eq!(vec![0; 0], *streams[0].dependencies);

    assert_eq!(vec![0, 1], *streams[0].instructions);
}

#[test]
fn reads_and_writes_of_same_operand_are_not_reordered() {
    let access = Access::Read;
    let prior_access = Access::Write;
    test_that_accesses_are_not_reordered(access, prior_access);
}

#[test]
fn writes_and_writes_of_same_operand_are_not_reordered() {
    let access = Access::Write;
    let prior_access = Access::Write;
    test_that_accesses_are_not_reordered(access, prior_access);
}

#[test]
fn writes_and_reads_of_same_operand_are_not_reordered() {
    let access = Access::Write;
    let prior_access = Access::Read;
    test_that_accesses_are_not_reordered(access, prior_access);
}

fn test_that_accesses_are_not_reordered(access: Access, prior_access: Access) {
    let instructions = get_test_instructions(None).unwrap();
    let instructions = Arc::new(instructions);
    let simple_instructions = make_simple_instructions(&instructions);
    let simple_instructions = Arc::new(simple_instructions);
    let expected_transactions = get_all_instruction_transactions(&simple_instructions);
    let expected_read_write_pairs =
        get_operand_transaction_pairs(&access, &prior_access, &expected_transactions);

    let minimum_write_before_read_for_new_stream = 4;
    let minimum_dependents_for_stream = 12;
    let minimum_stream_instructions = 32;
    let actual_streams = make_streams(
        &simple_instructions,
        minimum_write_before_read_for_new_stream,
        minimum_dependents_for_stream,
        minimum_stream_instructions,
    );
    let actual_streams = Arc::new(actual_streams);
    let max_concurrent_streams = 32;
    let actual_transactions = simulate_execution_and_collect_transactions(
        &actual_streams,
        &instructions,
        &simple_instructions,
        max_concurrent_streams,
    );
    let actual_read_write_pairs =
        get_operand_transaction_pairs(&access, &prior_access, &actual_transactions);

    for (operand, expected_pairs) in expected_read_write_pairs.iter() {
        let actual_pairs = actual_read_write_pairs.get(operand).unwrap();
        assert_eq!(expected_pairs, actual_pairs);
    }
}
