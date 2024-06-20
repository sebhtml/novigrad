use std::{collections::VecDeque, sync::Arc};
#[cfg(test)]
mod tests;

use crate::{
    stream::{DeviceStream, StreamTrait},
    streams::stream::Stream,
    tensor::Error,
    Device, Instruction,
};

use super::{SchedulerTrait, StreamEventHandler};

pub struct GpuStreamScheduler<Handler>
where
    Handler: StreamEventHandler + Send + Sync,
{
    device: Device,
    dependents: Vec<Vec<usize>>,
    initial_pending_dependencies: Vec<usize>,
    current_pending_dependencies: Vec<usize>,
    completed_logical_streams: usize,
    handler: Handler,
    streams: Arc<Vec<Stream>>,
    instructions: Arc<Vec<Instruction>>,
    queued_logical_streams: VecDeque<usize>,
    free_device_streams: VecDeque<DeviceStream>,
    used_device_streams: VecDeque<(usize, DeviceStream)>,
}

impl<Handler> SchedulerTrait<Handler> for GpuStreamScheduler<Handler>
where
    Handler: StreamEventHandler + Send + Sync + Clone,
{
    fn new(
        device: &Device,
        execution_units_len: usize,
        streams: &std::sync::Arc<Vec<Stream>>,
        handler: &Handler,
        instructions: &std::sync::Arc<Vec<Instruction>>,
    ) -> Self {
        println!("GPU Scheduler execution_units_len: {execution_units_len}");
        let pending_dependencies = streams.iter().map(|x| x.dependencies.len()).collect();
        let mut dependents = vec![vec![]; streams.len()];
        for (dependent, dependencies) in streams.iter().map(|x| &x.dependencies).enumerate() {
            for dependency in dependencies.iter() {
                dependents[*dependency].push(dependent);
            }
        }

        let free_device_streams = (0..execution_units_len)
            .map(|_| device.new_stream())
            .collect::<Result<VecDeque<DeviceStream>, _>>()
            .unwrap();
        Self {
            device: device.clone(),
            dependents,
            initial_pending_dependencies: pending_dependencies,
            current_pending_dependencies: Default::default(),
            completed_logical_streams: 0,
            handler: handler.clone(),
            streams: streams.clone(),
            instructions: instructions.clone(),
            queued_logical_streams: Default::default(),
            free_device_streams: free_device_streams,
            used_device_streams: Default::default(),
        }
    }

    fn start(&mut self) {}

    fn stop(&mut self) {}

    fn execute(&mut self) {
        self.completed_logical_streams = 0;
        self.current_pending_dependencies = self.initial_pending_dependencies.clone();
        // Queue immediately all logical streams with no dependencies.
        for (logical_stream, pending_dependencies) in
            self.current_pending_dependencies.iter().enumerate()
        {
            if *pending_dependencies == 0 {
                self.queued_logical_streams.push_back(logical_stream);
            }
        }

        // Launch initial logical streams on physical streams.
        while self.maybe_launch_device_stream().unwrap() {}

        // While any logical stream has not completed its execution.
        while self.completed_logical_streams != self.dependents.len() {
            let _ = self.wait_for_device_stream().unwrap();
            let _ = self.maybe_launch_device_stream().unwrap();
        }
    }
}

impl<Handler> GpuStreamScheduler<Handler>
where
    Handler: StreamEventHandler + Send + Sync + Clone,
{
    fn maybe_launch_device_stream(&mut self) -> Result<bool, Error> {
        // There is a queued logical stream and there is an available physical stream.
        if !self.queued_logical_streams.is_empty() && !self.free_device_streams.is_empty() {
            match (
                self.queued_logical_streams.pop_front(),
                self.free_device_streams.pop_front(),
            ) {
                (Some(logical_stream), Some(device_stream)) => {
                    device_stream.wait_for_default()?;
                    // The launch of a kernel on a GPU stream is asynchronous from the
                    // perspective of the host.
                    self.handler.on_execute(
                        &self.streams,
                        &self.instructions,
                        logical_stream,
                        &self.device,
                        &device_stream,
                    )?;
                    self.used_device_streams
                        .push_back((logical_stream, device_stream));
                    return Ok(true);
                }
                _ => panic!("Not supposed to happen"),
            }
        }
        Ok(false)
    }

    fn wait_for_device_stream(&mut self) -> Result<bool, Error> {
        match self.used_device_streams.pop_front() {
            Some((logical_stream, device_stream)) => {
                device_stream.wait_for()?;
                self.completed_logical_streams += 1;
                let dependents = &self.dependents[logical_stream];
                for dependent in dependents.iter() {
                    self.current_pending_dependencies[*dependent] -= 1;
                    let pending_dependencies = self.current_pending_dependencies[*dependent];
                    if pending_dependencies == 0 {
                        self.queued_logical_streams.push_back(*dependent);
                    }
                }
                self.free_device_streams.push_back(device_stream);
                Ok(true)
            }
            _ => panic!("Not supposed to happen bro"),
        }
    }
}
