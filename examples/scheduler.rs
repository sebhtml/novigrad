use novigrad::{
    mega_man_attention::get_megaman_attention_instructions,
    scheduler::{scheduler::Scheduler, InstructionEmitter},
    streams::{instruction::make_simple_instructions, stream::make_streams},
};
use std::sync::Arc;

fn main() {
    let instructions = get_megaman_attention_instructions().unwrap();
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
    let mut scheduler = Scheduler::new(execution_units_len, &streams, &handler, &instructions);

    let n = 100;
    scheduler.start();
    for _ in 0..n {
        scheduler.execute();
    }
    scheduler.stop();
}
