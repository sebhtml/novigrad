use std::{sync::Arc, thread::JoinHandle};

use crate::{
    schedulers::{SchedulerTrait, StreamEventHandler},
    streams::stream::Stream,
    tensor::Error,
    Device, Instruction,
};

use super::{controller::Controller, execution_unit::ExecutionUnit, queue::Queue, Command};

pub struct CpuStreamScheduler<Handler>
where
    Handler: StreamEventHandler + Send + Sync,
{
    #[allow(unused)]
    scheduler_command_queue: Arc<Queue<Command>>,
    controller_command_queue: Arc<Queue<Command>>,
    controller: Option<Controller>,
    execution_units: Option<Vec<ExecutionUnit<Handler>>>,
    controller_handle: Option<JoinHandle<Controller>>,
    execution_unit_handles: Option<Vec<JoinHandle<Result<ExecutionUnit<Handler>, Error>>>>,
}

impl<Handler> SchedulerTrait<Handler> for CpuStreamScheduler<Handler>
where
    Handler: StreamEventHandler + Clone + Send + Sync + 'static,
{
    fn new(
        device: &Device,
        maximum_device_streams: usize,
        streams: &Arc<Vec<Stream>>,
        handler: &Handler,
        instructions: &Arc<Vec<Instruction>>,
    ) -> Self {
        // Create structures

        // Various command queues.
        let scheduler_command_queue = Arc::new(Queue::default());
        let controller_command_queue = Arc::new(Queue::default());
        let execution_unit_command_queues = (0..maximum_device_streams)
            .map(|_| Arc::new(Queue::<Command>::default()))
            .collect::<Vec<_>>();

        // For the execution units.
        let execution_units = (0..maximum_device_streams)
            .map(|ordinal| {
                ExecutionUnit::new(
                    device,
                    ordinal,
                    &execution_unit_command_queues[ordinal],
                    &controller_command_queue,
                    handler.clone(),
                    streams,
                    instructions,
                )
            })
            .collect::<Vec<_>>();

        // For the controller.
        let controller = Controller::new(
            streams,
            &scheduler_command_queue,
            &controller_command_queue,
            &execution_unit_command_queues,
            maximum_device_streams,
        );
        Self {
            scheduler_command_queue,
            controller_command_queue,
            controller: Some(controller),
            execution_units: Some(execution_units),
            controller_handle: None,
            execution_unit_handles: None,
        }
    }

    /// Pre-conditions
    /// - self.execution_units is some
    /// - self.execution_unit_handles is none
    /// - self.controller is some
    /// - self.controller_handle is none
    /// Post-conditions
    /// - self.execution_units is none
    /// - self.execution_unit_handles is some
    /// - self.controller is none
    /// - self.controller_handle is some
    fn start(&mut self) {
        // Spawn threads
        let execution_unit_handles = self
            .execution_units
            .take()
            .unwrap()
            .into_iter()
            .map(|execution_unit| ExecutionUnit::spawn(execution_unit))
            .collect::<Vec<_>>();
        self.execution_unit_handles = Some(execution_unit_handles);

        let controller_handle = Controller::spawn(self.controller.take().unwrap());
        self.controller_handle = Some(controller_handle);
    }

    /// Pre-conditions
    /// - self.execution_units is none
    /// - self.execution_unit_handles is some
    /// - self.controller is none
    /// - self.controller_handle is some
    /// Post-conditions
    /// - self.execution_units is some
    /// - self.execution_unit_handles is none
    /// - self.controller is some
    /// - self.controller_handle is none
    fn stop(&mut self) {
        // Join the controller
        self.controller_command_queue.push_back(Command::Stop);
        let controller = self.controller_handle.take().unwrap().join().unwrap();
        self.controller = Some(controller);

        // Join the execution units.
        let execution_units = self
            .execution_unit_handles
            .take()
            .unwrap()
            .into_iter()
            .map(|x| x.join().unwrap())
            .collect::<Vec<_>>();
        self.execution_units = Some(execution_units.into_iter().map(|x| x.unwrap()).collect());
    }

    /// Execute all streams.
    fn execute(&mut self) {
        self.controller_command_queue.push_back(Command::Execute);
        let command = self.scheduler_command_queue.pop_front();
        match command {
            Some(Command::ExecutionCompletion) => {}
            _ => panic!(),
        }
    }
}
