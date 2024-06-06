use crate::{
    mega_man_attention::MegaManAttentionModel,
    neural_machine::streams::{print_instructions, print_streams},
    neural_program::NeuralProgram,
    tensor::Error,
    Adam, BinaryOperator, Category, Device, OptimizerTrait, SoftmaxCrossEntropyLoss, Tokenizer,
    TokenizerTrait, UnaryModel,
};
use std::{collections::BTreeMap, ops::Deref};
use test_case::test_case;

use super::{
    get_instruction_transactions, make_simple_instructions, make_streams, spawn_and_join_streams,
    Access, Transaction,
};

fn get_test_instructions(filter: Option<Category>) -> Result<Vec<(Vec<usize>, Vec<usize>)>, Error> {
    let device = Device::default();
    let tokenizer = Tokenizer::ascii_tokenizer();
    let vocab_size = tokenizer.vocab_size();
    let sequence_length = 32;
    let model = MegaManAttentionModel::new(&device, sequence_length, vocab_size)?;
    let model: Box<dyn UnaryModel> = Box::new(model);
    let loss_operator = SoftmaxCrossEntropyLoss::new(&device);
    let loss_operator: Box<dyn BinaryOperator> = Box::new(loss_operator);
    let learning_rate = 0.05;
    let optimizer = Adam::new(learning_rate, 0.9, 0.98, 1e-9);
    let optimizer: Box<dyn OptimizerTrait> = Box::new(optimizer);
    let program = NeuralProgram::try_new(&device, &model, &loss_operator, &optimizer)?;
    let instructions = program.instructions;
    let instructions = match filter {
        Some(category) => instructions
            .into_iter()
            .filter(|x| x.category() == category)
            .collect(),
        None => instructions,
    };
    let simple_instructions = make_simple_instructions(&instructions);
    Ok(simple_instructions)
}

#[test_case(None ; "no category filter")]
#[test_case(Some(Category::Inference) ; "inference filter")]
#[test_case(Some(Category::Loss) ; "loss filter")]
#[test_case(Some(Category::Gradient) ; "gradient filter")]
#[test_case(Some(Category::Optimization) ; "optimization filter")]
fn each_instruction_is_executed_exactly_once(filter: Option<Category>) {
    let instructions = get_test_instructions(filter).unwrap();
    let expected_instructions = (0..instructions.len()).collect::<Vec<_>>();
    let minimum_write_before_read_for_new_stream = 4;
    let minimum_stream_instructions = 32;
    let streams = make_streams(
        &instructions,
        minimum_write_before_read_for_new_stream,
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
    assert_eq!(2810, instructions.len());
    let minimum_write_before_read_for_new_stream = 4;
    let minimum_stream_instructions = 32;
    let streams = make_streams(
        &instructions,
        minimum_write_before_read_for_new_stream,
        minimum_stream_instructions,
    );
    let actual_instructions = streams
        .iter()
        .map(|i| i.instructions.deref().clone())
        .collect::<Vec<Vec<usize>>>()
        .concat();
    assert_eq!(2810, actual_instructions.len());
}

#[test]
fn the_streams_length_are_correct() {
    let instructions = get_test_instructions(None).unwrap();
    let minimum_write_before_read_for_new_stream = 4;
    let minimum_stream_instructions = 32;
    let streams = make_streams(
        &instructions,
        minimum_write_before_read_for_new_stream,
        minimum_stream_instructions,
    );
    print_streams("test", &streams);
    assert_eq!(
        106,
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
    let minimum_stream_instructions = 32;
    let streams = make_streams(
        &instructions,
        minimum_write_before_read_for_new_stream,
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
    let minimum_stream_instructions = 2;
    let streams = make_streams(
        &instructions,
        minimum_write_before_read_for_new_stream,
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
    let minimum_stream_instructions = 1;
    let streams = make_streams(
        &instructions,
        minimum_write_before_read_for_new_stream,
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
    let minimum_stream_instructions = 32;
    let streams = make_streams(
        &instructions,
        minimum_write_before_read_for_new_stream,
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
    let minimum_stream_instructions = 32;
    let streams = make_streams(
        &instructions,
        minimum_write_before_read_for_new_stream,
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
    let minimum_stream_instructions = 1;
    let streams = make_streams(
        &instructions,
        minimum_write_before_read_for_new_stream,
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
    let minimum_stream_instructions = 32;
    let streams = make_streams(
        &instructions,
        minimum_write_before_read_for_new_stream,
        minimum_stream_instructions,
    );

    print_streams("test", &streams);

    assert_eq!(1, streams.len());

    assert_eq!(vec![0; 0], *streams[0].dependencies);

    // TODO discard empty streams
    assert_eq!(vec![0, 1], *streams[0].instructions);
}

#[test_case(None ; "no category filter")]
#[test_case(Some(Category::Inference) ; "inference filter")]
#[test_case(Some(Category::Loss) ; "loss filter")]
#[test_case(Some(Category::Gradient) ; "gradient filter")]
#[test_case(Some(Category::Optimization) ; "optimization filter")]
fn reads_and_writes_of_same_operand_are_not_reordered(filter: Option<Category>) {
    let access = Access::Read;
    let prior_access = Access::Write;
    let instructions = get_test_instructions(filter).unwrap();
    let expected_transactions = get_all_instruction_transactions(&instructions);
    let expected_read_write_pairs =
        get_operand_transaction_pairs(&access, &prior_access, &expected_transactions);

    let minimum_write_before_read_for_new_stream = 4;
    let minimum_stream_instructions = 32;
    let actual_streams = make_streams(
        &instructions,
        minimum_write_before_read_for_new_stream,
        minimum_stream_instructions,
    );
    let actual_transactions = spawn_and_join_streams(&actual_streams, &instructions);
    let actual_read_write_pairs =
        get_operand_transaction_pairs(&access, &prior_access, &actual_transactions);

    for (operand, expected_pairs) in expected_read_write_pairs.iter() {
        let actual_pairs = actual_read_write_pairs.get(operand).unwrap();
        assert_eq!(expected_pairs, actual_pairs);
    }
}

#[test_case(None ; "no category filter")]
#[test_case(Some(Category::Inference) ; "inference filter")]
#[test_case(Some(Category::Loss) ; "loss filter")]
#[test_case(Some(Category::Gradient) ; "gradient filter")]
#[test_case(Some(Category::Optimization) ; "optimization filter")]
fn writes_and_writes_of_same_operand_are_not_reordered(filter: Option<Category>) {
    let access = Access::Write;
    let prior_access = Access::Write;
    let instructions = get_test_instructions(filter).unwrap();
    let expected_transactions = get_all_instruction_transactions(&instructions);
    let expected_read_write_pairs =
        get_operand_transaction_pairs(&access, &prior_access, &expected_transactions);

    let minimum_write_before_read_for_new_stream = 4;
    let minimum_stream_instructions = 32;
    let actual_streams = make_streams(
        &instructions,
        minimum_write_before_read_for_new_stream,
        minimum_stream_instructions,
    );
    let actual_transactions = spawn_and_join_streams(&actual_streams, &instructions);
    let actual_read_write_pairs =
        get_operand_transaction_pairs(&access, &prior_access, &actual_transactions);

    for (operand, expected_pairs) in expected_read_write_pairs.iter() {
        let actual_pairs = actual_read_write_pairs.get(operand).unwrap();
        assert_eq!(expected_pairs, actual_pairs);
    }
}

#[test_case(None ; "no category filter")]
#[test_case(Some(Category::Inference) ; "inference filter")]
#[test_case(Some(Category::Loss) ; "loss filter")]
#[test_case(Some(Category::Gradient) ; "gradient filter")]
#[test_case(Some(Category::Optimization) ; "optimization filter")]
fn writes_and_reads_of_same_operand_are_not_reordered(filter: Option<Category>) {
    let access = Access::Write;
    let prior_access = Access::Read;
    let instructions = get_test_instructions(filter).unwrap();
    let expected_transactions = get_all_instruction_transactions(&instructions);
    let expected_read_write_pairs =
        get_operand_transaction_pairs(&access, &prior_access, &expected_transactions);

    let minimum_write_before_read_for_new_stream = 4;
    let minimum_stream_instructions = 32;
    let actual_streams = make_streams(
        &instructions,
        minimum_write_before_read_for_new_stream,
        minimum_stream_instructions,
    );
    let actual_transactions = spawn_and_join_streams(&actual_streams, &instructions);
    let actual_read_write_pairs =
        get_operand_transaction_pairs(&access, &prior_access, &actual_transactions);

    for (operand, expected_pairs) in expected_read_write_pairs.iter() {
        let actual_pairs = actual_read_write_pairs.get(operand).unwrap();
        assert_eq!(expected_pairs, actual_pairs);
    }
}

fn get_all_instruction_transactions(instructions: &[(Vec<usize>, Vec<usize>)]) -> Vec<Transaction> {
    let mut transactions = vec![];
    for (instruction, (inputs, outputs)) in instructions.iter().enumerate() {
        let mut operand_transactions = get_instruction_transactions(instruction, inputs, outputs);
        transactions.extend_from_slice(&mut operand_transactions);
    }
    transactions
}

// Example: for each read, find the prior write.
// Basically there are read accesses and write accesses.
// Here are the 4 pillars of the memory model:
// - a read has a prior write and it must remain the same. Changing the prior write makes the result incorrect.
// - a write has a prior write and it must remain the same. Changing the prior write makes the result incorrect.
// - a write has a prior read and it must remain the same. Changing the prior read makes the result incorrect.
// - a read has a prior read and it can change. Changing the prior read is allowed.
//        Example, if instructions 1, 2, 3 read operand 44, all those orderings are valid ones:
//           - 1, 2, 3
//           - 3, 2, 1
//           - 2, 1, 3
//           - ...
//       If we have 12 attention heads, that means that we can have 12 concurrent streams.
fn get_operand_transaction_pairs(
    access: &Access,
    prior_access: &Access,
    transactions: &[Transaction],
) -> BTreeMap<usize, Vec<(Transaction, Transaction)>> {
    // Group transactions per operand.
    let operand_transactions = group_by_operand(transactions);
    // For each read of an operand, find the most recent write before it­.
    let mut operand_pairs = BTreeMap::<usize, Vec<(Transaction, Transaction)>>::new();
    for (operand, transactions) in operand_transactions.iter() {
        for i in 0..transactions.len() {
            let transaction_i = &transactions[i];
            if &transaction_i.access == access {
                // Find the most recent write to this operand that happened in the past.
                for j in (0..i).rev() {
                    let transaction_j = &transactions[j];
                    if &transaction_j.access == prior_access {
                        let pair = (transaction_i.to_owned(), transaction_j.to_owned());
                        operand_pairs.entry(*operand).or_default().push(pair);
                        break;
                    }
                }
            }
        }
    }
    for (_, pairs) in operand_pairs.iter_mut() {
        // The tests use == so sorting makes the tests pass.
        pairs.sort();
    }
    operand_pairs
}

fn group_by_operand(transactions: &[Transaction]) -> BTreeMap<usize, Vec<Transaction>> {
    let mut operand_transactions = BTreeMap::<usize, Vec<Transaction>>::new();
    for transaction in transactions.iter() {
        let operand = transaction.operand;
        operand_transactions
            .entry(operand)
            .or_default()
            .push(transaction.to_owned());
    }
    operand_transactions
}
