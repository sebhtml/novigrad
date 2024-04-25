== CUDA training MR ==

- in reset(), if new len is different, do a cuda reallocation
- use Result for return type of Device methods
- Don't call reset inside Tensor methods
- remove Tensor::set method
- remove Tensor::get method

== backward method MR ==

- add method backward in LearningTensor
- store device in LearningTensor
- remove back_propagation function

== Concat test MR ==

- implement Concat
- implement Matmul
- implement Add
- add a test with mega_man and cuda

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