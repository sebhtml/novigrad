use std::{collections::BTreeSet, ops::Deref};

use crate::{
    mega_man_attention::MegaManAttentionModel, neural_program::NeuralProgram, tensor::Error, Adam,
    BinaryOperator, Device, OptimizerTrait, SoftmaxCrossEntropyLoss, Tokenizer, TokenizerTrait,
    UnaryModel,
};

use super::{
    get_all_instruction_transactions, get_instruction_transactions, get_operand_transaction_pairs,
    make_simple_instructions, make_streams, Access, Stream, Transaction,
};

fn get_test_instructions() -> Result<Vec<(Vec<usize>, Vec<usize>)>, Error> {
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
    let simple_instructions = make_simple_instructions(&instructions);
    Ok(simple_instructions)
}

#[test]
fn each_instruction_is_executed_exactly_once() {
    let instructions = get_test_instructions().unwrap();
    let expected_instructions = (0..instructions.len()).collect::<Vec<_>>();
    let minimum_write_before_read_for_new_stream = 4;
    let streams = make_streams(&instructions, minimum_write_before_read_for_new_stream);
    let mut actual_instructions = streams
        .iter()
        .map(|x| x.instructions.deref().clone())
        .collect::<Vec<_>>()
        .concat();
    actual_instructions.sort();
    assert_eq!(expected_instructions, actual_instructions);
}

/// Simulate an execution of streams and emit operand transactions.
fn spawn_and_join_streams(
    streams: &[Stream],
    instructions: &[(Vec<usize>, Vec<usize>)],
) -> Vec<Transaction> {
    let mut actual_transactions = vec![];
    let mut unreached_streams = BTreeSet::<usize>::new();
    for i in 0..streams.len() {
        unreached_streams.insert(i);
    }
    let mut spawned_streams = BTreeSet::<usize>::new();
    let mut joined_streams = BTreeSet::<usize>::new();
    while joined_streams.len() != streams.len() {
        let mut stream_to_spawn: Option<usize> = None;
        // Find a stream that can be spawned.
        for unreached_stream in unreached_streams.iter() {
            let mut can_spawn = true;
            let dependencies = &streams[*unreached_stream].dependencies;
            for dependency in dependencies {
                if !joined_streams.contains(dependency) {
                    can_spawn = false;
                    break;
                }
            }
            if can_spawn {
                stream_to_spawn = Some(*unreached_stream);
                break;
            }
        }
        if let Some(stream_to_spawn) = stream_to_spawn {
            // Spawn it.
            unreached_streams.remove(&stream_to_spawn);
            spawned_streams.insert(stream_to_spawn);
            // Emit transactions on the execution unit pipeline.
            let stream_instructions = &streams[stream_to_spawn].instructions;
            for instruction in stream_instructions.iter() {
                let instruction = *instruction;
                let (inputs, outputs) = &instructions[instruction];
                let mut instruction_transactions =
                    get_instruction_transactions(instruction, inputs, outputs);
                actual_transactions.extend_from_slice(&mut instruction_transactions);
            }
            // Immediately join the thread.
            joined_streams.insert(stream_to_spawn);
        }
    }
    actual_transactions
}

#[test]
fn the_instructions_length_and_streams_length_are_correct() {
    let instructions = get_test_instructions().unwrap();
    assert_eq!(2810, instructions.len());
    let minimum_write_before_read_for_new_stream = 4;
    let streams = make_streams(&instructions, minimum_write_before_read_for_new_stream);
    assert_eq!(2797, streams.len());
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
    let streams = make_streams(&instructions, minimum_write_before_read_for_new_stream);

    assert_eq!(5, streams.len());

    assert_eq!(vec![0], *streams[0].instructions);
    assert_eq!(vec![1], *streams[1].instructions);
    assert_eq!(vec![2], *streams[2].instructions);
    assert_eq!(vec![3], *streams[3].instructions);
    assert_eq!(vec![4, 5], *streams[4].instructions);
}

#[test]
fn problem_2_for_streams() {
    let instructions = vec![
        (vec![0], vec![1]),
        (vec![1], vec![2]),
        (vec![1], vec![3]),
        (vec![1], vec![4]),
        (vec![1], vec![5]),
        (vec![2, 3, 4, 5], vec![6]),
    ];

    let minimum_write_before_read_for_new_stream = 4;
    let streams = make_streams(&instructions, minimum_write_before_read_for_new_stream);

    for (i, stream) in streams.iter().enumerate() {
        println!(
            "stream {},  dependencies {:?}  instructions {:?}",
            i, stream.dependencies, stream.instructions
        )
    }

    assert_eq!(6, streams.len());

    assert_eq!(vec![5], *streams[0].dependencies);
    assert_eq!(vec![5], *streams[1].dependencies);
    assert_eq!(vec![5], *streams[2].dependencies);
    assert_eq!(vec![5], *streams[3].dependencies);
    assert_eq!(vec![0, 1, 2, 3], *streams[4].dependencies);
    assert_eq!(vec![] as Vec<usize>, *streams[5].dependencies);

    assert_eq!(vec![1], *streams[0].instructions);
    assert_eq!(vec![2], *streams[1].instructions);
    assert_eq!(vec![3], *streams[2].instructions);
    assert_eq!(vec![4], *streams[3].instructions);
    assert_eq!(vec![5], *streams[4].instructions);
    assert_eq!(vec![0], *streams[5].instructions);
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
    let streams = make_streams(&instructions, minimum_write_before_read_for_new_stream);

    for (i, stream) in streams.iter().enumerate() {
        println!(
            "stream {},  dependencies {:?}  instructions {:?}",
            i, stream.dependencies, stream.instructions
        )
    }

    assert_eq!(6, streams.len());

    assert_eq!(vec![5], *streams[0].dependencies);
    assert_eq!(vec![5], *streams[1].dependencies);
    assert_eq!(vec![5], *streams[2].dependencies);
    assert_eq!(vec![5], *streams[3].dependencies);
    assert_eq!(vec![0, 1, 2, 3], *streams[4].dependencies);
    assert_eq!(vec![] as Vec::<usize>, *streams[5].dependencies);

    assert_eq!(vec![1], *streams[0].instructions);
    assert_eq!(vec![2], *streams[1].instructions);
    assert_eq!(vec![3], *streams[2].instructions);
    assert_eq!(vec![4], *streams[3].instructions);
    assert_eq!(vec![5, 6], *streams[4].instructions);
    assert_eq!(vec![0], *streams[5].instructions);
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
    let streams = make_streams(&instructions, minimum_write_before_read_for_new_stream);

    for (i, stream) in streams.iter().enumerate() {
        println!(
            "stream {},  dependencies {:?}  instructions {:?}",
            i, stream.dependencies, stream.instructions
        )
    }

    assert_eq!(12, streams.len());

    assert_eq!(vec![10], *streams[0].dependencies);
    assert_eq!(vec![10], *streams[1].dependencies);
    assert_eq!(vec![10], *streams[2].dependencies);
    assert_eq!(vec![10], *streams[3].dependencies);
    assert_eq!(vec![0, 1, 2, 3], *streams[4].dependencies);

    assert_eq!(vec![11], *streams[5].dependencies);
    assert_eq!(vec![11], *streams[6].dependencies);
    assert_eq!(vec![11], *streams[7].dependencies);
    assert_eq!(vec![11], *streams[8].dependencies);
    assert_eq!(vec![5, 6, 7, 8], *streams[9].dependencies);

    assert_eq!(vec![] as Vec::<usize>, *streams[10].dependencies);
    assert_eq!(vec![4], *streams[11].dependencies);

    assert_eq!(vec![1], *streams[0].instructions);
    assert_eq!(vec![2], *streams[1].instructions);
    assert_eq!(vec![3], *streams[2].instructions);
    assert_eq!(vec![4], *streams[3].instructions);
    assert_eq!(vec![5], *streams[4].instructions);
    assert_eq!(vec![7], *streams[5].instructions);
    assert_eq!(vec![8], *streams[6].instructions);
    assert_eq!(vec![9], *streams[7].instructions);
    assert_eq!(vec![10], *streams[8].instructions);
    assert_eq!(vec![11], *streams[9].instructions);
    assert_eq!(vec![0], *streams[10].instructions);
    assert_eq!(vec![6], *streams[11].instructions);
}

#[test]
fn reads_and_writes_of_same_operand_are_not_reordered() {
    let access = Access::Read;
    let prior_access = Access::Write;
    let instructions = get_test_instructions().unwrap();
    let expected_transactions = get_all_instruction_transactions(&instructions);
    let expected_read_write_pairs =
        get_operand_transaction_pairs(&access, &prior_access, &expected_transactions);

    let minimum_write_before_read_for_new_stream = 99;
    let actual_streams = make_streams(&instructions, minimum_write_before_read_for_new_stream);
    let actual_transactions = spawn_and_join_streams(&actual_streams, &instructions);
    let actual_read_write_pairs =
        get_operand_transaction_pairs(&access, &prior_access, &actual_transactions);

    for (operand, expected_pairs) in expected_read_write_pairs.iter() {
        let actual_pairs = actual_read_write_pairs.get(operand).unwrap();
        assert_eq!(expected_pairs, actual_pairs);
    }
}

#[test]
fn writes_and_writes_of_same_operand_are_not_reordered() {
    let access = Access::Write;
    let prior_access = Access::Write;
    let instructions = get_test_instructions().unwrap();
    let expected_transactions = get_all_instruction_transactions(&instructions);
    let expected_read_write_pairs =
        get_operand_transaction_pairs(&access, &prior_access, &expected_transactions);

    let minimum_write_before_read_for_new_stream = 4;
    let actual_streams = make_streams(&instructions, minimum_write_before_read_for_new_stream);
    let actual_transactions = spawn_and_join_streams(&actual_streams, &instructions);
    let actual_read_write_pairs =
        get_operand_transaction_pairs(&access, &prior_access, &actual_transactions);

    for (operand, expected_pairs) in expected_read_write_pairs.iter() {
        let actual_pairs = actual_read_write_pairs.get(operand).unwrap();
        assert_eq!(expected_pairs, actual_pairs);
    }
}

#[test]
fn writes_and_reads_of_same_operand_are_not_reordered() {
    let access = Access::Write;
    let prior_access = Access::Read;
    let instructions = get_test_instructions().unwrap();
    let expected_transactions = get_all_instruction_transactions(&instructions);
    let expected_read_write_pairs =
        get_operand_transaction_pairs(&access, &prior_access, &expected_transactions);

    let minimum_write_before_read_for_new_stream = 4;
    let actual_streams = make_streams(&instructions, minimum_write_before_read_for_new_stream);
    let actual_transactions = spawn_and_join_streams(&actual_streams, &instructions);
    let actual_read_write_pairs =
        get_operand_transaction_pairs(&access, &prior_access, &actual_transactions);

    for (operand, expected_pairs) in expected_read_write_pairs.iter() {
        let actual_pairs = actual_read_write_pairs.get(operand).unwrap();
        assert_eq!(expected_pairs, actual_pairs);
    }
}
