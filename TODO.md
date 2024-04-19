== Backlog ==
- store interior of Tensor in Rc
- optimizer should not receive tape
- add capability for having N inputs in forward method
- implement OperatorTrait for LossFunction
- enable tape recording only during training
- add Blas implementation for CuBlas using https://crates.io/crates/cudarc

- implement Dropout
- implement Concat
- implement Matmul
- implement Mask
- move learning rate in dataset details
- shuffle examples in each epoch
- implement TransformerBlock

- bpe tokenizer
- add gelu
- add tape to decouple compute from storage
- centralize panic! calls
- check if it's easy to cudaMalloc and cudaMemCpy and cudaFree
- add Blas implementation for AMD GPUs.