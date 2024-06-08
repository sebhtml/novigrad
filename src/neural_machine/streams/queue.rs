use std::{
    collections::VecDeque,
    sync::{Condvar, Mutex},
};

pub struct Queue<T> {
    deque: Mutex<VecDeque<T>>,
    cvar: Condvar,
}

impl<T> Default for Queue<T> {
    fn default() -> Self {
        Self {
            deque: Default::default(),
            cvar: Default::default(),
        }
    }
}

impl<T> Queue<T> {
    pub fn push_back(&self, item: T) {
        self.deque.lock().unwrap().push_back(item);
        self.cvar.notify_one();
    }

    pub fn pop_front(&self) -> Option<T> {
        let guard = self.deque.lock().unwrap();
        let mut guard = self.cvar.wait(guard).unwrap();
        guard.pop_front()
    }
}
