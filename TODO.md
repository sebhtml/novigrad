- clip grad norn after backward
- rename ARC prize to colored_mosaic_puzzles
- remove buffer store that allow reuse
- add option to print total_next_token_perplexity in TensorPrinter
- Implement MSELoss correctly using ReduceSum
- Implement SoftmaxCrossEntropyLoss correctly using other operators

== Manual mini-batch ==

- fix backward code of reduce sum square and cross-entropy loss
- use batch aggregated loss to compute gradient

== correct mini-batch implementation ==

- impement mini-batch in the model input tensor shape
- implement mini batch using broadcasting in the operators

== Story: Transformer batching ==

- increase examples in transformer test from 30 to 100
- use batching in transformer dataset
- use 4 layers in transformer model
- set maximum_incorrect_predicted_next_tokens to 0 in transformer dataset

== Story: Mega-man transformer ==

- use Mega_man.txt for transformer dataset

- re-add method zero_grad


== Story: use device pointer mode ==

- use device pointer mode for Gemm's alpha and beta (maybe this is the cause of pthread_rwlock_unlock)

== Story: gradient accumulation ==

- honour requires_grad() when updating gradients
- Make sure that backward instruction add on top of existing gradients (no overwrite)

== Story: Arc prize ==

- have one unified set for instructions, streams, scheduler instead of four (inference, loss, gradient, optimization)
- Implement Transformer idea for the Arc prize challenge (left-to-right residual connections)

- investigate performance issue with tons of call to pthread_rwlock_unlock

== Clean-up ==

- implement RMSNorm : https://arxiv.org/pdf/1910.07467
- simplify code that push gradient_instruction instructions (too much re-mapping of inputs to outputs)

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
