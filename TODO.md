- add tokenizer with encode and decode

== New operators ==

- implement Concat
- implement Matmul
- implement Add

- get rid of TrainWorkingMemory

== Initialization ==

- revisit initialization of weights and biases in Linear
- revisit initialization of embedding_table in Embedding

== Training ==

- in LearningTensor, gradient is None if requires_grad is false
- enable tape recording only during training
- implement mini batch
- move learning rate in dataset details
- shuffle examples in each epoch

== New operators (part 2) ==

- implement Dropout
- implement Mask
- implement Gelu
- implement TransformerBlock

== Tokens ==

- bpe tokenizer

== Refactoring ==

- determine the value of using_cross_entropy_loss at run time
- Don't call reset inside Tensor methods
- add rc device in tensor
- remove device argument in OperatorTrait
- store device in LearningTensor

== Devices ==

- add Blas implementation for AMD GPUs.
- support for google tpu
- support for apple metal
