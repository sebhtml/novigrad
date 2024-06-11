use std::{sync::Arc, thread::JoinHandle};

use crate::{streams::stream::Stream, Instruction};

use super::{
    controller::Controller, execution_unit::ExecutionUnit, queue::Queue, Command,
    StreamEventHandler,
};

pub struct Scheduler<Handler>
where
    Handler: StreamEventHandler + Send + Sync,
{
    #[allow(unused)]
    scheduler_command_queue: Arc<Queue<Command>>,
    controller_command_queue: Arc<Queue<Command>>,
    controller: Option<Controller>,
    execution_units: Option<Vec<ExecutionUnit<Handler>>>,
    controller_handle: Option<JoinHandle<Controller>>,
    execution_unit_handles: Option<Vec<JoinHandle<ExecutionUnit<Handler>>>>,
}

impl<Handler> Scheduler<Handler>
where
    Handler: StreamEventHandler + Clone + Send + Sync + 'static,
{
    pub fn new(
        execution_units_len: usize,
        streams: &Arc<Vec<Stream>>,
        handler: &Handler,
        instructions: &Arc<Vec<Instruction>>,
    ) -> Self {
        // Create structures

        // Various command queues.
        let scheduler_command_queue = Arc::new(Queue::default());
        let controller_command_queue = Arc::new(Queue::default());
        let execution_unit_command_queues = (0..execution_units_len)
            .map(|_| Arc::new(Queue::<Command>::default()))
            .collect::<Vec<_>>();

        // For the execution units.
        let execution_units = (0..execution_units_len)
            .map(|ordinal| {
                let execution_unit = ExecutionUnit::new(
                    ordinal,
                    &execution_unit_command_queues[ordinal],
                    &controller_command_queue,
                    handler.clone(),
                    streams,
                    instructions,
                );
                execution_unit
            })
            .collect::<Vec<_>>();

        // For the controller.
        let controller = Controller::new(
            streams,
            &scheduler_command_queue,
            &controller_command_queue,
            &execution_unit_command_queues,
            execution_units_len,
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
    pub fn start(&mut self) {
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
    pub fn stop(&mut self) {
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
        self.execution_units = Some(execution_units);
    }

    pub fn execute(&mut self) {
        self.controller_command_queue.push_back(Command::Execute);
        let command = self.scheduler_command_queue.pop_front();
        match command {
            Some(Command::ExecutionCompletion) => {}
            _ => panic!(),
        }
    }
}
