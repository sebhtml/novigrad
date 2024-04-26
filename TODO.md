== CUDA training ==

- cuda tests
- use Result for return type of Device methods

== New operators ==

- implement Concat
- implement Matmul
- implement Add

== Training ==

- implement mini batch
- enable tape recording only during training
- move learning rate in dataset details
- shuffle examples in each epoch

== Operators ==

- implement Dropout
- implement Mask
- implement Gelu
- implement TransformerBlock

== Tokens ==

- bpe tokenizer

== Refactoring ==

- determine the value of using_cross_entropy_loss at run time
- centralize panic! calls
- Don't call reset inside Tensor methods
- add rc device in tensor
- store device in LearningTensor

== Devices ==

- add Blas implementation for AMD GPUs.
- support for google tpu
- support for apple metal
