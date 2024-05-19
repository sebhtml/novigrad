- remove dyn Operator from OpCode
- delete trait Operator

-----------------

- Use Add instead of AddBackward
- rename Scale to ScalarMul
- SigmoidBackward and SoftmaxBackward are Mul
- simplify execute method arguments
- move execute methods outside of operators inside neural_machine
- simplify train.rs to have at most 1 call to infer, loss, backward, step() per example per epoch.

== GPT-1 Transformer ==

- implement Dropout
- implement Gelu
- implement LayerNorm
- implement Transformer

== Logging ==

- don't print machine on boot for chatbot
- in chatbot example, use special token end_of_text to disable loss for that unknown expected token

- print instruction category in log

== Cuda ==

- implement a mul cuda kernel

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
- remove DeviceEnum
- restore simple Errors (no line etc.)

== Fixes ==

- make list of things that are using Tensorf32::set_value
- remove random calls to unwrap()
- return ErrNoGradient if output tensor has no gradient

== Mini Batch ==

- implement mini batch
- image transformer

== Datasets ==

- CIFAR-10
- MNIST

== Program ==

- implement parallel execution of certain branches in parallel using a execution_group_id

== Import / Export ==

- serialize and deserialize model to ONNX format

== Positional Encoding ==

- implement positional encoding

== AMD ==

- Add support AMD GPUs (ROCm/rocBLAS) -> https://docs.rs/simt_rocblas_sys/latest/simt_rocblas_sys/struct.rocblas.html


== Devices ==

- Add support for Jim Keller's https://tenstorrent.com/cards/
- Add support for Google TPU
- Add support for Apple Metal
- Add support for Intel Arc
