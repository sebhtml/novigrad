== Attention ==

- implement Matmul
- implement Mask
- implement Scale
- implement Attention
- use Attention in megaman

== Fixes ==

- don't use Reshape in Megaman
- print number of parameters optimized by optimizer
- decouple tensor and device
- don't backward if last
- enable tape recording only during training

== Others ==

- implement Add

- avoid re-allocating output tensors every time
- replace Box dyn by enum
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

== Devices ==

- add Blas implementation for AMD GPUs.
- support for google tpu
- support for apple metal
