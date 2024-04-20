== Optimizer PR
- optimizer should not receive tape
- enable tape recording only during training
- move learning rate in dataset details

== Tensor storage PR ==
- store interior of Tensor in Rc

== Backlog ==
- add Blas implementation for CuBlas using https://crates.io/crates/cudarc

- implement Dropout
- implement Concat
- implement Matmul
- implement Mask
- add Gelu

- shuffle examples in each epoch
- implement TransformerBlock

- bpe tokenizer

- add tape to decouple compute from storage
- centralize panic! calls
- check if it's easy to cudaMalloc and cudaMemCpy and cudaFree
- add Blas implementation for AMD GPUs.