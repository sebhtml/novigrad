use std::sync::Arc;

use crate::{
    datasets::mega_man_multi_head_attention::get_multi_head_attention_model_instructions,
    schedulers::simulate_execution_and_collect_instructions,
    streams::{instruction::make_simple_instructions, stream::make_streams},
    Device,
};

use super::{
    simulate_execution_and_collect_transactions,
    transaction::{
        get_all_instruction_transactions, get_operand_transaction_pairs, Access, TransactionEmitter,
    },
    InstructionEmitter, SchedulerTrait,
};

pub fn verify_that_accesses_are_not_reordered<Scheduler>(access: Access, prior_access: Access)
where
    Scheduler: SchedulerTrait<TransactionEmitter>,
{
    let device = Device::default();
    let instructions = get_multi_head_attention_model_instructions(&device).unwrap();
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
    let maximum_device_streams = 32;
    let actual_transactions = simulate_execution_and_collect_transactions::<Scheduler>(
        &device,
        &actual_streams,
        &instructions,
        &simple_instructions,
        maximum_device_streams,
    );
    let actual_read_write_pairs =
        get_operand_transaction_pairs(&access, &prior_access, &actual_transactions);

    for (operand, expected_pairs) in expected_read_write_pairs.iter() {
        let actual_pairs = actual_read_write_pairs.get(operand).unwrap();
        assert_eq!(expected_pairs, actual_pairs);
    }
}

pub fn verify_that_all_instructions_are_executed_with_out_of_order_execution<Scheduler>()
where
    Scheduler: SchedulerTrait<InstructionEmitter>,
{
    let device = Device::default();
    let instructions = get_multi_head_attention_model_instructions(&device).unwrap();
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
    let maximum_device_streams = 32;

    let executed_instructions = simulate_execution_and_collect_instructions::<Scheduler>(
        &device,
        &actual_streams,
        &instructions,
        maximum_device_streams,
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

pub fn verify_that_all_instructions_are_executed_in_each_scheduler_execution<Scheduler>()
where
    Scheduler: SchedulerTrait<InstructionEmitter>,
{
    let device = Device::default();
    let instructions = get_multi_head_attention_model_instructions(&device).unwrap();
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
    let maximum_device_streams = 32;

    let handler = InstructionEmitter::new();
    let mut scheduler = Scheduler::new(
        &device,
        maximum_device_streams,
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
