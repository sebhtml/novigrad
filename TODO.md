== Fixes ==

- avoid re-allocating output tensors every time (allocate them in Architecture)
- does it work without gradient clipping ?
- remove parameters from struct architecture
- print number of parameters optimized by optimizer
- use variables in mega_man architecture

== Attention ==

- implement Scale
- implement Mask
- implement Attention
- use Attention in megaman

== Fixes ==

- decouple tensor and device
- don't backward if last
- enable backward tape recording only during training

== Mini Batch ==

- implement mini batch
- shuffle examples in each epoch

== Fixes ==

- don't use Reshape in Megaman

== Parallel Execution ==

- add forward tape
- implement parallel execution of certain branches in parallel using a execution_group_id

== Datasets ==

- put load_dataset in a polymorphic enum
- put txt file in a data directory (check rust documentation)

== Initialization ==

- revisit initialization of weights and biases in Linear
- revisit initialization of embedding_table in Embedding

== Multi-Head Attention ==

- implement Concat
- implement MultiHeadAttention

== GPT-1 Transformer ==

- implement Dropout
- implement Add
- implement Gelu
- implement LayerNorm
- implement Transformer

== Positional Encoding ==

- implement positional encoding

== Refactoring ==

- remove tensor f32 matmul
- determine the value of using_cross_entropy_loss at run time
- add rc device in tensor
- remove device argument in OperatorTrait
- store device in LearningTensor

== Devices ==

- add Blas implementation for AMD GPUs.
- support for Google tpu
- support for Apple metal
