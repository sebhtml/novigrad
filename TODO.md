== gradient MR ==

- store output tensor in operator
- store gradients in output tensors during back prop
- remove Clone for Tensor
- in reset(), if new len is different, do a cuda reallocation
- use Result for return type of Device methods
- don't store output in tape
- add a test with simple and cuda
- add a test with mega_man and cuda

== Concat MR ==

- implement Concat
- implement Matmul
- implement Add

== training MR ==
- enable tape recording only during training
- move learning rate in dataset details

== Backlog ==
- implement Dropout
- implement Mask
- implement Gelu
- implement TransformerBlock

- shuffle examples in each epoch
- bpe tokenizer

- determine the value of using_cross_entropy_loss at run time
- centralize panic! calls
- check if it's easy to cudaMalloc and cudaMemCpy and cudaFree
- add Blas implementation for AMD GPUs.