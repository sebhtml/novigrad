- fix ScaleBackward alpha use
- remove optimizer

== Tensor clean-up ==

- device.tensor should take size instead of rows, cols

== Refactoring ==

- models with no operator fields
- merge the many load_examples functions
- remove DatasetEnum
- move code from training/mod.rs to training/train.rs
- remove DeviceEnum
- restore simple Errors (no line etc.)

== Fixes ==

- models return expected size in Error instead of inputsize and outputsize
- make list of things that are using Tensorf32::setvalue
- remove recycle
- remove forward from Operator trait
- remove random calls to unwrap()
- return ErrNoGradient if output tensor has no gradient
- add function to print a program instructions, inputs, output

== Multi-Head Attention ==

- implement MultiHeadAttention

== Mini Batch ==

- implement mini batch
- shuffle examples in each epoch

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

== Refactoring ==

- determine the value of using_cross_entropy_loss at run time

== Devices ==

- use cuda stream to realize a tensor
- implement a mul cuda kernel
- Add support for Jim Keller's https://tenstorrent.com/cards/
- Add support AMD GPUs (ROCm/rocBLAS)
- Add support for Google TPU
- Add support for Apple Metal
- Add support for Intel Arc
