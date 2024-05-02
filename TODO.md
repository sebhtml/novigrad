== Attention ==

- implement Scale
- implement Mask
- implement Attention
- use Attention in megaman

== Fixes ==

- use variables in mega_man architecture
- put load_dataset in a polymorphic enum
- print number of parameters optimized by optimizer
- decouple tensor and device
- don't backward if last
- enable tape recording only during training
- don't use Reshape in Megaman

== Others ==

- implement Add

- avoid re-allocating output tensors every time (allocate them in Architecture)
- put txt file in a data directory (check rust documentation)

== Initialization ==

- revisit initialization of weights and biases in Linear
- revisit initialization of embedding_table in Embedding

== Training ==

- implement mini batch
- shuffle examples in each epoch

== Multi-Head Attention ==

- implement Concat
- implement MultiHeadAttention

== Dropout ==

- implement Dropout
- implement Gelu

== Transformer ==

- implement FeedForward
- implement AddAndNorm
- implement Transformer

== Positional encoding ==

- implement positional encoding

== Refactoring ==

- determine the value of using_cross_entropy_loss at run time
- add rc device in tensor
- remove device argument in OperatorTrait
- store device in LearningTensor
- use MatMul::forward in Embedding and in Linear
- use MatMul::backward in Embedding and in Linear
- replace Box dyn by enum

== Devices ==

- add Blas implementation for AMD GPUs.
- support for google tpu
- support for apple metal
