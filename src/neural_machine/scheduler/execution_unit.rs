use std::{
    sync::Arc,
    thread::{self, JoinHandle},
};

use crate::{
    error,
    stream::DeviceStream,
    streams::stream::Stream,
    tensor::{Error, ErrorEnum},
    Device, DeviceTrait, Instruction,
};

use super::{queue::Queue, Command, StreamEventHandler};

/// https://en.wikipedia.org/wiki/Instruction_pipelining
pub struct ExecutionUnit<Handler>
where
    Handler: StreamEventHandler + Send + Sync,
{
    #[allow(unused)]
    device: Device,
    _ordinal: usize,
    handler: Handler,
    streams: Arc<Vec<Stream>>,
    instructions: Arc<Vec<Instruction>>,
    execution_unit_command_queue: Arc<Queue<Command>>,
    controller_command_queue: Arc<Queue<Command>>,
    completed_items: usize,
}

impl<Handler> ExecutionUnit<Handler>
where
    Handler: StreamEventHandler + Send + Sync + 'static,
{
    pub fn new(
        device: &Device,
        ordinal: usize,
        execution_unit_command_queue: &Arc<Queue<Command>>,
        controller_command_queue: &Arc<Queue<Command>>,
        handler: Handler,
        streams: &Arc<Vec<Stream>>,
        instructions: &Arc<Vec<Instruction>>,
    ) -> Self {
        Self {
            device: device.clone(),
            _ordinal: ordinal,
            handler,
            streams: streams.clone(),
            instructions: instructions.clone(),
            execution_unit_command_queue: execution_unit_command_queue.clone(),
            controller_command_queue: controller_command_queue.clone(),
            completed_items: 0,
        }
    }

    pub fn spawn(mut execution_unit: Self) -> JoinHandle<Result<Self, Error>> {
        let handle = thread::spawn(|| {
            let device_stream = execution_unit
                .device
                .stream()
                .map_err(|_| error!(ErrorEnum::UnsupportedOperation))?;
            while execution_unit.step(&device_stream) {}
            Ok(execution_unit)
        });
        handle
    }

    fn step(&mut self, device_stream: &DeviceStream) -> bool {
        // Fetch
        let command = self.execution_unit_command_queue.pop_front();
        match command {
            Some(Command::Stop) => {
                return false;
            }
            Some(Command::WorkUnitDispatch(stream)) => {
                // Call handler to execute the instructions for that stream.
                self.handler
                    .on_execute(&self.streams, &self.instructions, stream, device_stream)
                    .unwrap();
                // Writeback
                self.controller_command_queue
                    .push_back(Command::WorkUnitCompletion(stream));
                self.completed_items += 1;
            }
            _ => {}
        }
        true
    }
}
