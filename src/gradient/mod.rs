use std::{borrow::Borrow, cell::RefCell, ops::Deref, rc::Rc};

use crate::{accelerator::Accelerator, DeltaWorkingMemory, Error, Tensor};

mod tape;
pub use tape::*;
mod differentiable_tensor;
pub use differentiable_tensor::*;
mod differentiable_module_enum;
pub use differentiable_module_enum::*;
mod back_propagation;
pub use back_propagation::*;

pub trait DifferentiableModuleTrait {
    fn compute_gradient(
        &mut self,
        accelerator: &Accelerator,
        layer_input: &Tensor,
        layer_output_delta: &Tensor,
    );

    fn commit_change(&mut self, accelerator: &Accelerator, learning_rate: f32)
        -> Result<(), Error>;

    fn forward(
        &mut self,
        accelerator: &Accelerator,
        layer_input: &Tensor,
        layer_output: &mut Tensor,
    ) -> Result<(), Error>;

    // TODO backward should return Error
    fn backward(
        &self,
        accelerator: &Accelerator,
        layer_output_delta: &Tensor,
        previous_layer_output_delta: &mut Tensor,
    );

    // TODO get_layer_delta should return Error
    fn get_layer_output_delta(
        &self,
        accelerator: &Accelerator,
        working_memory: &mut DeltaWorkingMemory,
        layer_input: &Tensor,
        layer_output: &Tensor,
        back_propagated_layer_output_delta: &Tensor,
        is_last_layer: bool,
        layer_output_delta: &mut Tensor,
    );
}

pub trait Forward {
    fn forward(&mut self, layer_input: &Tensor) -> Result<Tensor, Error>;
    fn accelerator(&self) -> Rc<Accelerator>;
    fn tape(&self) -> Rc<RefCell<Tape>>;
}

pub struct DifferentiableModule {
    accelerator: Rc<Accelerator>,
    tape: Rc<RefCell<Tape>>,
    variant: Rc<RefCell<DifferentiableModuleEnum>>,
}

impl Forward for DifferentiableModule {
    fn forward(&mut self, layer_input: &Tensor) -> Result<Tensor, Error> {
        let mut layer_output = Tensor::default();
        let variant = &mut *self.variant.deref().borrow_mut();
        variant.forward(self.accelerator.deref(), layer_input, &mut layer_output)?;
        self.tape.deref().borrow_mut().push(
            self.variant.borrow(),
            Rc::new(layer_output.clone()).borrow(),
        );
        Ok(layer_output)
    }

    fn accelerator(&self) -> Rc<Accelerator> {
        self.accelerator.clone()
    }

    fn tape(&self) -> Rc<RefCell<Tape>> {
        self.tape.clone()
    }
}

impl DifferentiableModule {
    pub fn new(
        accelerator: Rc<Accelerator>,
        tape: Rc<RefCell<Tape>>,
        variant: Rc<RefCell<DifferentiableModuleEnum>>,
    ) -> Self {
        Self {
            accelerator,
            tape,
            variant,
        }
    }
    pub fn compute_gradient(&mut self, layer_input: &Tensor, layer_output_delta: &Tensor) {
        let variant = &mut *self.variant.deref().borrow_mut();
        variant.compute_gradient(self.accelerator.deref(), layer_input, layer_output_delta)
    }

    pub fn commit_change(&mut self, learning_rate: f32) -> Result<(), Error> {
        let variant = &mut *self.variant.deref().borrow_mut();
        variant.commit_change(self.accelerator.deref(), learning_rate)
    }

    pub fn backward(&self, layer_output_delta: &Tensor, previous_layer_output_delta: &mut Tensor) {
        let variant = &mut *self.variant.deref().borrow_mut();
        variant.backward(
            self.accelerator.deref(),
            layer_output_delta,
            previous_layer_output_delta,
        )
    }

    pub fn get_layer_output_delta(
        &self,
        working_memory: &mut DeltaWorkingMemory,
        layer_input: &Tensor,
        layer_output: &Tensor,
        back_propagated_layer_output_delta: &Tensor,
        is_last_layer: bool,
        layer_output_delta: &mut Tensor,
    ) {
        let variant = &mut *self.variant.deref().borrow_mut();
        variant.get_layer_output_delta(
            self.accelerator.deref(),
            working_memory,
            layer_input,
            layer_output,
            back_propagated_layer_output_delta,
            is_last_layer,
            layer_output_delta,
        )
    }
}

pub struct FullDifferentiableModuleConfig<'a> {
    pub accelerator: &'a Rc<Accelerator>,
    pub tape: &'a Rc<RefCell<Tape>>,
    pub config: &'a DifferentiableModuleConfig,
}

impl<'a> Into<DifferentiableModule> for &FullDifferentiableModuleConfig<'a> {
    fn into(self) -> DifferentiableModule {
        let config = self.config;
        let variant = config.into();
        let accelerator = self.accelerator;
        let tape = self.tape;
        DifferentiableModule {
            accelerator: accelerator.clone(),
            tape: tape.clone(),
            variant: Rc::new(RefCell::new(variant)),
        }
    }
}
