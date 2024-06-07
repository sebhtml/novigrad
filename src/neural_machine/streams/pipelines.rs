use std::{cell::RefCell, collections::LinkedList, ops::Deref, rc::Rc, sync::Arc};

use crate::Instruction;

use super::{scheduler::StreamEventHandler, stream::Stream};

pub struct Controller {
    dependents: Vec<Vec<usize>>,
    pending_dependencies: Vec<usize>,
    emit_pipeline: Rc<RefCell<LinkedList<usize>>>,
    retire_pipeline: Rc<RefCell<LinkedList<usize>>>,
}

impl Controller {
    pub fn new(
        streams: &[Stream],
        emit_pipeline: &Rc<RefCell<LinkedList<usize>>>,
        retire_pipeline: &Rc<RefCell<LinkedList<usize>>>,
    ) -> Self {
        //println!("streams {:?}", streams.len());
        let pending_dependencies = streams.iter().map(|x| x.dependencies.len()).collect();
        //println!("pending_dependencies {:?}", pending_dependencies);
        let mut dependents = vec![vec![]; streams.len()];
        for (dependent, dependencies) in streams.iter().map(|x| &x.dependencies).enumerate() {
            for dependency in dependencies.iter() {
                dependents[*dependency].push(dependent);
            }
        }
        Self {
            dependents,
            pending_dependencies,
            emit_pipeline: emit_pipeline.clone(),
            retire_pipeline: retire_pipeline.clone(),
        }
    }

    pub fn start(&self) {
        // Emit immediately all streams with no dependencies.
        for (stream, _) in self.pending_dependencies.iter().enumerate() {
            self.maybe_emit(stream);
        }
    }

    fn maybe_emit(&self, stream: usize) {
        let pending_dependencies = self.pending_dependencies[stream];
        //println!("stream {} pending_dependencies {}", stream, pending_dependencies);
        if pending_dependencies == 0 {
            //println!("EMIT {}", stream);
            self.emit_pipeline.deref().borrow_mut().push_back(stream);
        }
    }

    pub fn step(&mut self) -> bool {
        if let Some(stream) = self.retire_pipeline.deref().borrow_mut().pop_front() {
            //RETIRE {}", stream);
            let dependents = &self.dependents[stream];
            //println!("stream {}  dependents {:?}", stream, dependents);
            for dependent in dependents.iter() {
                self.pending_dependencies[*dependent] -= 1;
                self.maybe_emit(*dependent);
            }
            true
        } else {
            false
        }
    }
}

impl Drop for Controller {
    fn drop(&mut self) {
        //println!("pending_dependencies {:?}", self.pending_dependencies);
    }
}

pub struct ExecutionUnit<Handler: StreamEventHandler> {
    handler: Rc<RefCell<Handler>>,
    streams: Arc<Vec<Stream>>,
    instructions: Arc<Vec<Instruction>>,
    emit_pipeline: Rc<RefCell<LinkedList<usize>>>,
    retire_pipeline: Rc<RefCell<LinkedList<usize>>>,
}

impl<Handler: StreamEventHandler + Clone> ExecutionUnit<Handler> {
    pub fn new(
        emit_pipeline: &Rc<RefCell<LinkedList<usize>>>,
        retire_pipeline: &Rc<RefCell<LinkedList<usize>>>,
        handler: &Rc<RefCell<Handler>>,
        streams: &Arc<Vec<Stream>>,
        instructions: &Arc<Vec<Instruction>>,
    ) -> Self {
        Self {
            handler: handler.clone(),
            streams: streams.clone(),
            instructions: instructions.clone(),
            emit_pipeline: emit_pipeline.clone(),
            retire_pipeline: retire_pipeline.clone(),
        }
    }

    pub fn step(&mut self) -> bool {
        if let Some(stream) = self.emit_pipeline.deref().borrow_mut().pop_front() {
            // Call handler to execute the instructions for that stream.
            self.handler
                .deref()
                .borrow_mut()
                .on_execute(&self.streams, &self.instructions, stream)
                .unwrap();
            // Retire the stream.
            self.retire_pipeline.deref().borrow_mut().push_back(stream);
            true
        } else {
            false
        }
    }
}
