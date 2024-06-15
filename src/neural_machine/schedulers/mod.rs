use std::sync::Arc;

use crate::{cpu_scheduler::StreamEventHandler, streams::stream::Stream, Device, Instruction};

pub trait SchedulerTrait<Handler>
where
    Handler: StreamEventHandler,
{
    fn new(
        device: &Device,
        execution_units_len: usize,
        streams: &Arc<Vec<Stream>>,
        handler: &Handler,
        instructions: &Arc<Vec<Instruction>>,
    ) -> Self;

    fn start(&mut self);

    fn stop(&mut self);

    fn execute(&mut self);
}
