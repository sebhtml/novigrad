- remove module config structs
- enable tape recording only during training
- add Blas implementation for CuBlas
- add capability for having N blocks side-by-side in a layer (required for multi-head attention)

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