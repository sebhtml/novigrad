== backward MR ==

- rewrite back propagation function
- group compute_gradients, get_layer_output_delta with backward in OperatorTrait

== cuda MR ==
- add Blas implementation for CuBlas using https://crates.io/crates/cudarc

== training MR ==
- enable tape recording only during training
- move learning rate in dataset details

== Backlog ==
- implement Dropout
- implement Concat
- implement Matmul
- implement Mask
- implement Gelu
- implement TransformerBlock

- shuffle examples in each epoch
- bpe tokenizer

- determine the value of using_cross_entropy_loss at run time
- centralize panic! calls
- check if it's easy to cudaMalloc and cudaMemCpy and cudaFree
- add Blas implementation for AMD GPUs.