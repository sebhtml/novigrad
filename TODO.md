- SigmoidBackward and SoftmaxBackward are Mul
- remove ReshapeBackward
- Use Add instead of AddBackward
- remove dyn Operator from OpCode
- delete trait Operator
- remove word Backward
- print instruction category in log

--------------------------------
- use macro for errors

- make sure that all OpCode have >= 2 inputs
- no values in OpCode, put them insteast in OpCodeArguments

---------------------

- increase learning rates
- try Tensor<f8>
- Split Softmax in Exp + other operators to reuse them.

-----------------

- simplify train.rs to have at most 1 call to infer, loss, backward, step() per example per epoch.

== Regularization ==

- implement Dropout

== GPT-1 Transformer ==

- implement Gelu
- implement LayerNorm
- implement Transformer

- Div
- Pow

== Logging ==

- investigate performance issue with tons of call to pthread_rwlock_unlock
- don't print machine on boot for chatbot
- in chatbot example, use special token end_of_text to disable loss for that unknown expected token

- simplify Loss

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
