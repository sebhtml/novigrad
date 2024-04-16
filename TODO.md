- in backprop, move the computation of gradients to gradient/
- add accelerator in module with Forward trait
- use imperative form for architecture
- add Blas implementation for CuBlas
- add capability for having N blocks side-by-side in a layer (required for multi-head attention)

- implement Dropout
- move learning rate in dataset details
- shuffle examples in each epoch
- implement transformer

- bpe tokenizer
- add gelu
- add tape to decouple compute from storage
- centralize panic! calls
- check if it's easy to cudaMalloc and cudaMemCpy and cudaFree
- add Blas implementation for AMD GPUs.