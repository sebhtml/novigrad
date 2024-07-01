== Story: Adam fixes ==

- adam t should be in 0..num_iterations
- use adam in transformer dataset

== Story: no buffer store ==

- remove buffer store that allow reuse

== Story: use device pointer mode ==

- use device pointer mode for Gemm's alpha and beta (maybe this is the cause of pthread_rwlock_unlock)

== Story: gradient accumulation ==

- honour requires_grad() when updating gradients
- Make sure that backward instruction add on top of existing gradients (no overwrite)

== Story: Transformer bug hunting ==

- use 4 layers in transformer model
- increase examples in transformer test from 10 to 100

== Story: Arc prize ==

- have one unified set for instructions, streams, scheduler instead of four (inference, loss, gradient, optimization)
- Implement Transformer idea for the Arc prize challenge (left-to-right residual connections)


- simplify code that push gradient_instruction instructions (too much re-mapping of inputs to outputs)
- investigate performance issue with tons of call to pthread_rwlock_unlock
- maybe the pthread lock is caused by checking if the loss is 0

== Clean-up ==

- remove all calls to set_values
- rewrite ResidualSumOfSquares using CUDA
- implement Transpose with CUDA

- refactor gelu, sigmoid, gelu_derivative in cpu module
- move ./src/devices/cuda/tests.rs tests that are not related to cuda to ./src/devices/tests.rs
- move ./src/devices/cpu/tests.rs tests that are not related to cpu to ./src/devices/tests.rs

== Story: AMD ROCm with HIP ==

- Add support AMD GPUs (ROCm/rocBLAS) -> https://docs.rs/simt_rocblas_sys/latest/simt_rocblas_sys/struct.rocblas.html

== Backlog ==

---------------

- implement ArgMax operator https://onnx.ai/onnx/operators/onnx__ArgMax.html
- rename RowMax to ArgMax (https://onnx.ai/onnx/operators/onnx__ArgMax.html)
- add code that discard useless instructions, for example when a operand write is never read betfore the next write

---------------

- improve Bernoulli CUDA kernel by using other shift values for halt the indices

- add Tensor categories
- use Category::Constant to determine constants
- use Category::Parameter to determine parameters

---------------

- device interface use <T>
- Implement code with f16

---------------------

- implement Conv2D

== Performance ==

- simplify train.rs to have at most 1 call to infer, loss, compute_gradient, optimize() per example per epoch.


== Other things ==

- investigate calls to Device::tensor_f32

== Tensor clean-up ==

- device.tensor should take size instead of rows, cols

== Refactoring ==

- merge the many load_examples / generate_examples functions

== Fixes ==

- remove random calls to unwrap()
- return ErrNoGradient if output tensor has no gradient

== Mini Batch ==

- implement mini batch
- image transformer

== Datasets ==

- CIFAR-10
- MNIST

== Import / Export ==

- serialize and deserialize model to ONNX format

== Positional Encoding ==

- implement positional encoding

== Devices ==

- Add support for Jim Keller's https://tenstorrent.com/cards/
- Add support for Google TPU
- Add support for Apple Metal
- Add support for Intel Arc
