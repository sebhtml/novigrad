- implement realize() in the forward pass
- does it work without gradient clipping ?
- print number of parameters optimized by optimizer
- in TensorF32, replace rows, cols by dims
- decouple tensor and device
- remove tensor f32 matmul
- merge network/mod.rs + train.rs -> training/mod.rs

== Attention ==

- implement Scale
- implement Mask
- implement Attention
- use Attention in megaman

== Mini Batch ==

- implement mini batch
- shuffle examples in each epoch

== Fixes ==

- don't use Reshape in Megaman

== Parallel Execution ==

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

- determine the value of using_cross_entropy_loss at run time

== Devices ==

- add Blas implementation for AMD GPUs.
- support for Google tpu
- support for Apple metal
