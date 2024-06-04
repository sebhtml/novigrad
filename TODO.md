- spawn threads with CPU device
- add device stream support in devices to execute attention heads in parallel

== Transformer ==

- implement ArgMax operator https://onnx.ai/onnx/operators/onnx__ArgMax.html
- add Standardize

- implement LayerNormalization which is Standardize + ScalarMul + Add
- implement Gelu
- implement LayerNorm
- implement Transformer
- increase examples in mega_man_attention from 10 to 100

---------------

- add code that discard useless instructions, for example when a operand write is never read betfore the next write
- investigate performance issue with tons of call to pthread_rwlock_unlock

- improve Bernoulli CUDA kernel by using other shift values for halt the indices

- add Tensor categories
- use Category::Constant to determine constants
- use Category::Parameter to determine parameters

---------------
- TODO adam t should be in 0..num_iterations
- don't break during training when loss reaches 0.0

- use Attributes for Gemm, see https://onnx.ai/onnx/intro/concepts.html
- remove all calls to set_values
- rewrite ResidualSumOfSquares using CUDA

- device interface use <T>
- Implement code with f16

---------------------

- make sure that all OpCode have >= 2 inputs
- implement Conv2D

== Performance ==

- simplify train.rs to have at most 1 call to infer, loss, compute_gradient, optimize() per example per epoch.

== AMD ==

- Add support AMD GPUs (ROCm/rocBLAS) -> https://docs.rs/simt_rocblas_sys/latest/simt_rocblas_sys/struct.rocblas.html

- Make sure that backward instruction add on top of existing gradients (no overwrite)

== Other things ==

- investigate calls to Device::tensor_f32

== Tensor clean-up ==

- device.tensor should take size instead of rows, cols

== Refactoring ==

- merge the many load_examples functions

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
