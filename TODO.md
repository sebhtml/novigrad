== Attention ==

- implement Scale
- implement Mask
- implement Attention
- use Attention in megaman

== Mini Batch ==

- implement mini batch
- shuffle examples in each epoch

== Fixes ==

- don't use Reshape in Megaman to reduce the number of model parameters

== Parallel Execution ==

- implement parallel execution of certain branches in parallel using a execution_group_id

== Datasets ==

- put load_dataset in a polymorphic enum

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

- Add support AMD GPUs (ROCm/rocBLAS)
- Add support for Google TPU
- Add support for Apple Metal
- Add support for Intel Arc