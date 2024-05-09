== Multi-Head Attention ==

- implement Concat
- implement MultiHeadAttention

- remove random calls to unwrap()

== Mini Batch ==

- implement mini batch
- shuffle examples in each epoch

== Parallel Execution ==

- implement parallel execution of certain branches in parallel using a execution_group_id

== Datasets ==

- put load_dataset in a polymorphic enum

== Initialization ==

- revisit initialization of weights and biases in Linear
- revisit initialization of embedding_table in Embedding

== GPT-1 Transformer ==

- implement Dropout
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