use std::{borrow::Borrow, cell::RefCell, ops::Deref, rc::Rc};

use crate::{accelerator::Accelerator, DeltaWorkingMemory, Error, Tensor};

mod tape;
pub use tape::*;
mod differentiable_module_enum;
pub use differentiable_module_enum::*;

pub struct DifferentiableTensor {
    pub tensor: Tensor,
    pub gradient: Tensor,
    pub has_gradient: bool,
}

impl DifferentiableTensor {
    pub fn new(tensor: Tensor) -> Self {
        Self {
            tensor,
            gradient: Default::default(),
            has_gradient: Default::default(),
        }
    }
    pub fn commit_change(&mut self, accelerator: &Accelerator, learning_rate: f32) {
        if !self.has_gradient {
            return;
        }

        let op_result = Tensor::saxpy(
            accelerator,
            -learning_rate,
            &self.gradient,
            &mut self.tensor,
        );
        op_result.expect("Ok");
        self.has_gradient = false;
    }
}

impl From<Tensor> for DifferentiableTensor {
    fn from(value: Tensor) -> Self {
        Self::new(value)
    }
}

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

pub struct DifferentiableModule {
    tape: Rc<RefCell<Tape>>,
    variant: Rc<RefCell<DifferentiableModuleEnum>>,
}

impl DifferentiableModuleTrait for DifferentiableModule {
    fn compute_gradient(
        &mut self,
        accelerator: &Accelerator,
        layer_input: &Tensor,
        layer_output_delta: &Tensor,
    ) {
        let variant = &mut *self.variant.deref().borrow_mut();
        variant.compute_gradient(accelerator, layer_input, layer_output_delta)
    }

    fn commit_change(
        &mut self,
        accelerator: &Accelerator,
        learning_rate: f32,
    ) -> Result<(), Error> {
        let variant = &mut *self.variant.deref().borrow_mut();
        variant.commit_change(accelerator, learning_rate)
    }

    fn forward(
        &mut self,
        accelerator: &Accelerator,
        layer_input: &Tensor,
        layer_output: &mut Tensor,
    ) -> Result<(), Error> {
        let variant = &mut *self.variant.deref().borrow_mut();
        variant.forward(accelerator, layer_input, layer_output)?;
        self.tape.deref().borrow_mut().push(
            self.variant.borrow(),
            Rc::new(layer_output.clone()).borrow(),
        );
        Ok(())
    }

    fn backward(
        &self,
        accelerator: &Accelerator,
        layer_output_delta: &Tensor,
        previous_layer_output_delta: &mut Tensor,
    ) {
        let variant = &mut *self.variant.deref().borrow_mut();
        variant.backward(accelerator, layer_output_delta, previous_layer_output_delta)
    }

    fn get_layer_output_delta(
        &self,
        accelerator: &Accelerator,
        working_memory: &mut DeltaWorkingMemory,
        layer_input: &Tensor,
        layer_output: &Tensor,
        back_propagated_layer_output_delta: &Tensor,
        is_last_layer: bool,
        layer_output_delta: &mut Tensor,
    ) {
        let variant = &mut *self.variant.deref().borrow_mut();
        variant.get_layer_output_delta(
            accelerator,
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
    pub tape: &'a Rc<RefCell<Tape>>,
    pub config: &'a DifferentiableModuleConfig,
}

impl<'a> Into<DifferentiableModule> for &FullDifferentiableModuleConfig<'a> {
    fn into(self) -> DifferentiableModule {
        let config = self.config;
        let variant = config.into();
        let tape = self.tape;
        DifferentiableModule {
            tape: tape.clone(),
            variant: Rc::new(RefCell::new(variant)),
        }
    }
}
