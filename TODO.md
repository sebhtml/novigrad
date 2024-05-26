- fix adam
- remove calls to set_values outside of tests

- fix precision in cuda dot kernel test with large size
- fix code when feature "cuda" is disabled.
- rewrite CrossEntropyLoss using Sum
- rewrite ResidualSumOfSquares using Sum

---------------------

- don't break during training when loss reaches 0.0

== Transformer ==

- add Standardize
- implement LayerNormalization which is Standardize + ScalarMul + Add
- implement Gelu
- implement LayerNorm
- implement Transformer
- increase examples in mega_man_attention from 10 to 100

---------------
- use const* f32 instead of &Tensor in DeviceInterface
- TODO adam t should be in 0..num_iterations
- add device stream support in devices to execute attention heads in parallel

- investigate performance issue with tons of call to pthread_rwlock_unlock

- device interface use <T>
- Implement code with f16

---------------------

- make sure that all OpCode have >= 2 inputs
- implement Conv2D
- no values in OpCode, put them instead in OpCodeArguments

== Performance ==

---------------------

- simplify train.rs to have at most 1 call to infer, loss, compute_gradient, optimize() per example per epoch.

== AMD ==

- Add support AMD GPUs (ROCm/rocBLAS) -> https://docs.rs/simt_rocblas_sys/latest/simt_rocblas_sys/struct.rocblas.html

- Make sure that backward instruction add on top of existing gradients (no overwrite)

== Initialization ==

- use Kaiming uniform initialization
- revisit initialization of weights and biases in Linear
- revisit initialization of embedding_table in Embedding

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
