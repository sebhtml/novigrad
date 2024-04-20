== Tensor storage PR ==
- use Rc<Tensor> in inputs and output of OperatorTrait

== Backlog ==
- enable tape recording only during training
- move learning rate in dataset details

- add Blas implementation for CuBlas using https://crates.io/crates/cudarc

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