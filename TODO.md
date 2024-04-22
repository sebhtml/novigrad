== CUDA MR ==

- use Result for return type of Device methods
- create tensors with device
- store Tensor on Device Cuda

== gradient MR ==

- store output tensor in operator
- store gradients in output tensors during back prop

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