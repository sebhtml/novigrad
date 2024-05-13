# Novigrad: A forward pass is all you need

An animal can do training and inference every day of its existence until the day of its death.

Usually, neural networks like LLMs use the forward-backward algorithm during training, and a forward pass during inference.
In 2022, Prof. Geoffrey Hinton invented the [forward-forward algorithm](https://arxiv.org/abs/2212.13345).
His work is the inspiration for Novigrad.
With Novigrad  ("New Town"), a forward pass is all you need, for both training and inference.

In AI, LLMs do their training once with a lot of data. Once this is done, they are deemed ready for inference.
At this point, they no longer learn. They look like they remember memories from the past. But that's simply because
we provide them with a context buffer of things of the past in an auto-regressive way.
LLMs remember the things that they saw during training. But they don't really do it during inference.

A true neural machine should do training and do inference, always. There is no difference between those activities.
Both training and inference are just simulating the future.

Novigrad is a system that compiles a neural network model to neural instructions.
These neural instructions are the basis of neural machines.

There is no backward pass. A forward pass is all you need.
To achieve the goal, the computing machinery required to do the things that usually occur during the backward pass is simply baked in the generated neural machine. A consequence of this is that resulting neural networks have the computing machinery for training and inference.
Animal brains are probably like that.

Novigrad aims to be a minimalist neural network framework written in Rust.

# Project goals

- Design and implement a neural system framework that generates neural networks that can always train and infer. (most animals do that everyday)
- Implement from scratch in Rust the [GPT-1](https://en.wikipedia.org/wiki/GPT-1) architecture (117 million parameters).
- Teach an AI to read and write by interacting with a user on a daily basis.

# Other frameworks

Rust
- [Burn](https://github.com/tracel-ai/burn/tree/main)
- [candle](https://github.com/huggingface/candle)
- [dfdx]( https://github.com/coreylowman/dfdx)
- [tch-rs](https://github.com/LaurentMazare/tch-rs)
- [wonnx](https://github.com/webonnx/wonnx/)

Python
- [PyTorch](https://github.com/pytorch/pytorch)
- [TensorFlow](https://github.com/tensorflow/tensorflow)
- [micrograd](https://github.com/karpathy/micrograd)
- [tinygrad](https://github.com/tinygrad/tinygrad)
- [teenygrad](https://github.com/tinygrad/teenygrad)

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