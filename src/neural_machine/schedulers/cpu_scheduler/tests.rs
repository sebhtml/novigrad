use std::sync::Arc;

use crate::{
    mega_man_attention::get_megaman_attention_instructions,
    neural_machine::streams::{instruction::make_simple_instructions, stream::make_streams},
    schedulers::{
        simulate_execution_and_collect_instructions, simulate_execution_and_collect_transactions, transaction::{get_all_instruction_transactions, get_operand_transaction_pairs, Access}, InstructionEmitter, SchedulerTrait
    },
    Device,
};

use super::scheduler::CpuStreamScheduler;

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
    let device = Device::default();
    let instructions = get_megaman_attention_instructions(&device).unwrap();
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
    let execution_units_len = 32;
    let actual_transactions = simulate_execution_and_collect_transactions(
        &device,
        &actual_streams,
        &instructions,
        &simple_instructions,
        execution_units_len,
    );
    let actual_read_write_pairs =
        get_operand_transaction_pairs(&access, &prior_access, &actual_transactions);

    for (operand, expected_pairs) in expected_read_write_pairs.iter() {
        let actual_pairs = actual_read_write_pairs.get(operand).unwrap();
        assert_eq!(expected_pairs, actual_pairs);
    }
}

#[test]
fn all_instructions_are_executed_with_out_of_order_execution() {
    let device = Device::default();
    let instructions = get_megaman_attention_instructions(&device).unwrap();
    let instructions = Arc::new(instructions);
    let simple_instructions = make_simple_instructions(&instructions);
    let simple_instructions = Arc::new(simple_instructions);

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
    let execution_units_len = 32;

    let executed_instructions = simulate_execution_and_collect_instructions(
        &device,
        &actual_streams,
        &instructions,
        execution_units_len,
    );

    // Same length
    assert_eq!(instructions.len(), executed_instructions.len());

    // Out-of-order execution means that the order is different.
    let sequential_instructions = (0..instructions.len()).collect::<Vec<_>>();
    assert_ne!(sequential_instructions, executed_instructions);

    // When sorted, the instructions are the same.
    let mut sorted_executed_instructions = executed_instructions;
    sorted_executed_instructions.sort();
    assert_eq!(sequential_instructions, sorted_executed_instructions);
}

#[test]
fn all_instructions_are_executed_in_each_scheduler_execution() {
    let device = Device::default();
    let instructions = get_megaman_attention_instructions(&device).unwrap();
    let instructions = Arc::new(instructions);
    let simple_instructions = make_simple_instructions(&instructions);
    let simple_instructions = Arc::new(simple_instructions);

    let minimum_write_before_read_for_new_stream = 4;
    let minimum_dependents_for_stream = 12;
    let minimum_stream_instructions = 32;
    let streams = make_streams(
        &simple_instructions,
        minimum_write_before_read_for_new_stream,
        minimum_dependents_for_stream,
        minimum_stream_instructions,
    );
    let streams = Arc::new(streams);
    let execution_units_len = 32;

    let handler = InstructionEmitter::new();
    let mut scheduler = CpuStreamScheduler::new(
        &device,
        execution_units_len,
        &streams,
        &handler,
        &instructions,
    );
    scheduler.start();

    let sequential_instructions = (0..instructions.len()).collect::<Vec<_>>();

    let n = 10;
    for _ in 0..n {
        scheduler.execute();
        let executed_instructions = &mut handler.executed_instructions.lock().unwrap();

        // Same length
        assert_eq!(instructions.len(), executed_instructions.len());

        // Out-of-order execution means that the order is different.
        assert_ne!(sequential_instructions, **executed_instructions);

        // When sorted, the instructions are the same.
        executed_instructions.sort();
        assert_eq!(sequential_instructions, **executed_instructions);

        // Clear the instructions.
        executed_instructions.clear();
    }
    scheduler.stop();
    //panic!()
}
