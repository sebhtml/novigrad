# Project goals

- Implement from scratch in Rust the [GPT-1](https://en.wikipedia.org/wiki/GPT-1) architecture (117 million parameters).

# Novigrad

Novigrad is a automatic differentiation engine with a backward (reverse) mode and a forward mode.
It aims to be a minimalist neural network framework written in Rust.

# Other frameworks

Rust
- [Burn](https://github.com/tracel-ai/burn/tree/main)
- [candle](https://github.com/huggingface/candle)
- [dfdx]( https://github.com/coreylowman/dfdx)
- [tch-rs](https://github.com/LaurentMazare/tch-rs)

Python
- [PyTorch](https://github.com/pytorch/pytorch)
- [TensorFlow](https://github.com/tensorflow/tensorflow)
- [micrograd](https://github.com/karpathy/micrograd)
- [teenygrad](https://github.com/tinygrad/teenygrad)
- [tinygrad](https://github.com/tinygrad/tinygrad)

C++
- [onnx](https://github.com/onnx/onnx)

# Hardware used

- [NVIDIA GeForce RTX 4060 Laptop GPU](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4060-4060ti/)
- Lenovo Legion 7 Laptop
- [AMD Ryzen 7 7840HS](https://www.amd.com/en/products/apu/amd-ryzen-7-7840hs)
- [AMD Radeon 780M Graphics](https://www.techpowerup.com/gpu-specs/radeon-780m.c4020)

# Roadmap

see [TODO.md](TODO.md)

# Run the program

```bash
cargo run --release
```

# Run the tests

```bash
cargo test --release
```

# Mega_man

Mega_man.txt comes from Wikipedia .
Text is available under the Creative Commons Attribution-ShareAlike License 4.0

# General Links

- [A Simple Introduction to Broadcasting](https://medium.com/@hunter-j-phillips/)a-simple-introduction-to-broadcasting-db8e581368b3
- [Mega man](https://en.wikipedia.org/wiki/Mega_Man)
- [Prof. Geoffrey Hinton - "Will digital intelligence replace biological intelligence?" Romanes Lecture](https://www.youtube.com/watch?v=N1TEjTeQeg0)

# Performance Links

- [Fast Multidimensional Matrix Multiplication on CPU from Scratch](https://siboehm.com/articles/22/Fast-MMM-on-CPU)

# Mathematics Links

- [Training Hidden Units: The Generalized Delta Rule](https://web.stanford.edu/group/pdplab/originalpdphandbook/Chapter%205.pdf)
- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [GAUSSIAN ERROR LINEAR UNITS (GELUS)](https://arxiv.org/pdf/1606.08415.pdf)
- [What is Cross-Entropy Loss: LLMs Explained](https://www.chatgptguide.ai/2024/03/03/what-is-cross-entropy-loss-llms-explained/)
- [Deriving categorical cross entropy and softmax](https://shivammehta25.github.io/posts/deriving-categorical-cross-entropy-and-softmax/)

# PyTorch Links

- [LINEAR](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- [Introducnn.Linear in PyTorch: Clearly Explainedtion](https://docs.kanaries.net/topics/Python/nn-linear)
- [Word Embeddings: Encoding Lexical Semantics](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html)

# CUDA links

- [Programming Tensor Cores in CUDA 9](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
- [gemm](https://docs.rs/cublas/latest/cublas/struct.API.html#method.gemm)
- [arrayfire::CublasMathMode](https://arrayfire.org/arrayfire-rust/arrayfire/enum.CublasMathMode.html)
- [cuBLAS](https://docs.nvidia.com/cuda/cublas/)