- chore: write loss gradient to loss
- TensorF32 in instructions
- use tensorf32 in operator trait

- Bake optimizer instructions in neural machine

- move concat and unconcat code to functions
- copy -> copy_from
- remove recycle

- Clip must preserve the direction of the tensor

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

== Datasets ==

- serialize and deserialize model to ONNX format

== Initialization ==

- revisit initialization of weights and biases in Linear
- revisit initialization of embedding_table in Embedding

== GPT-1 Transformer ==

- implement Dropout
- implement Gelu
- implement LayerNorm
- implement Transformer

== Positional Encoding ==

- implement positional encoding

== Devices ==

- use cuda stream to realize a tensor (is this useful ? CUDA execution is async by default)
- implement a mul cuda kernel
- Add support for Jim Keller's https://tenstorrent.com/cards/
- Add support AMD GPUs (ROCm/rocBLAS)
- Add support for Google TPU
- Add support for Apple Metal
- Add support for Intel Arc
