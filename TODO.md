- bpe tokenizer

== New operators ==

- implement Concat
- implement Matmul
- implement Add

== Initialization ==

- revisit initialization of weights and biases in Linear
- revisit initialization of embedding_table in Embedding

== Training ==

- enable tape recording only during training
- implement mini batch
- shuffle examples in each epoch

== New operators (part 2) ==

- implement Dropout
- implement Mask
- implement Gelu
- implement TransformerBlock

== Refactoring ==

- determine the value of using_cross_entropy_loss at run time
- add rc device in tensor
- remove device argument in OperatorTrait
- store device in LearningTensor

== Devices ==

- add Blas implementation for AMD GPUs.
- support for google tpu
- support for apple metal
