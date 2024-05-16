== Instructions that use TensorF32 ==

- use tensorf32 in operator trait
- TensorF32 in instructions

- Make sure that backward instruction add on top of existing gradients (no overwrite)
- Bake optimizer instructions in neural machine
- make clip_norm a parameter
- diag in mask to -inf

== Initialization ==

- use Kaiming uniform initialization
- revisit initialization of weights and biases in Linear
- revisit initialization of embedding_table in Embedding

== Clean-up ==

- Remove TensorF32 matmul
- remoze zero and name from tensor
- sigmoidGrad and softmaxBackward are Mul
- remove most of the Backward ops

== Things ==

- rename DatasetDetails to ModelTrainingDetails
- backward has no parameters

== Other things ==

- investigate calls to Device::tensor_f32
- test if Zero is really needed

- copy -> copy_from
- remove recycle

== Tensor clean-up ==

- device.tensor should take size instead of rows, cols

== Refactoring ==

- merge the many load_examples functions
- remove DatasetEnum
- move code from training/mod.rs to training/train.rs
- remove DeviceEnum
- restore simple Errors (no line etc.)

== Fixes ==

- make list of things that are using Tensorf32::set_value
- remove random calls to unwrap()
- return ErrNoGradient if output tensor has no gradient

== Mini Batch ==

- implement mini batch
- shuffle examples in each epoch

- image transformer

== Datasets ==

- CIFAR-10
- MNIST

== Program ==

- implement parallel execution of certain branches in parallel using a execution_group_id


== GPT-1 Transformer ==

- implement Dropout
- implement Gelu
- implement LayerNorm
- implement Transformer

== Positional Encoding ==

- implement positional encoding

== AMD ==

- Add support AMD GPUs (ROCm/rocBLAS) -> https://docs.rs/simt_rocblas_sys/latest/simt_rocblas_sys/struct.rocblas.html

== Datasets ==

- serialize and deserialize model to ONNX format

== Devices ==

- use cuda stream to realize a tensor (is this useful ? CUDA execution is async by default)
- implement a mul cuda kernel
- Add support for Jim Keller's https://tenstorrent.com/cards/
- Add support for Google TPU
- Add support for Apple Metal
- Add support for Intel Arc
