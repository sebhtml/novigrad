use std::{
    sync::Arc,
    thread::{self, JoinHandle},
};

use crate::streams::stream::Stream;

use super::{queue::Queue, Command};

pub struct Controller {
    dependents: Vec<Vec<usize>>,
    initial_pending_dependencies: Vec<usize>,
    current_pending_dependencies: Vec<usize>,
    scheduler_command_queue: Arc<Queue<Command>>,
    controller_command_queue: Arc<Queue<Command>>,
    execution_unit_command_queues: Vec<Arc<Queue<Command>>>,
    completed_streams: usize,
    execution_units_len: usize,
}

impl Controller {
    pub fn new(
        streams: &[Stream],
        scheduler_command_queue: &Arc<Queue<Command>>,
        controller_command_queue: &Arc<Queue<Command>>,
        execution_unit_command_queues: &Vec<Arc<Queue<Command>>>,
        execution_units_len: usize,
    ) -> Self {
        let pending_dependencies = streams.iter().map(|x| x.dependencies.len()).collect();
        let mut dependents = vec![vec![]; streams.len()];
        for (dependent, dependencies) in streams.iter().map(|x| &x.dependencies).enumerate() {
            for dependency in dependencies.iter() {
                dependents[*dependency].push(dependent);
            }
        }
        Self {
            dependents,
            initial_pending_dependencies: pending_dependencies,
            current_pending_dependencies: Default::default(),
            scheduler_command_queue: scheduler_command_queue.clone(),
            controller_command_queue: controller_command_queue.clone(),
            execution_unit_command_queues: execution_unit_command_queues.clone(),
            completed_streams: 0,
            execution_units_len,
        }
    }

    pub fn spawn(mut controller: Self) -> JoinHandle<Self> {
        
        thread::spawn(|| {
            while controller.step() {}
            controller
        })
    }

    fn maybe_dispatch(&self, stream: usize) {
        let pending_dependencies = self.current_pending_dependencies[stream];
        if pending_dependencies == 0 {
            let ordinal = stream % self.execution_units_len;
            self.execution_unit_command_queues[ordinal]
                .push_back(Command::WorkUnitDispatch(stream));
        }
    }

    pub fn step(&mut self) -> bool {
        let command = self.controller_command_queue.pop_front();
        match command {
            Some(Command::Execute) => {
                self.completed_streams = 0;
                self.current_pending_dependencies = self.initial_pending_dependencies.clone();
                // Dispatch immediately all streams with no dependencies.
                for (stream, _) in self.current_pending_dependencies.iter().enumerate() {
                    self.maybe_dispatch(stream);
                }
            }
            Some(Command::WorkUnitCompletion(stream)) => {
                self.completed_streams += 1;
                let dependents = &self.dependents[stream];
                for dependent in dependents.iter() {
                    self.current_pending_dependencies[*dependent] -= 1;
                    self.maybe_dispatch(*dependent);
                }
                if self.completed_streams == self.dependents.len() {
                    self.scheduler_command_queue
                        .push_back(Command::ExecutionCompletion);
                }
            }
            Some(Command::Stop) => {
                for ordinal in 0..self.execution_units_len {
                    self.execution_unit_command_queues[ordinal].push_back(Command::Stop);
                }
                return false;
            }
            _ => {}
        }
        true
    }
}
