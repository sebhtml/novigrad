# Run the program

```bash
cargo run --release
```

# Mega_man

Mega_man.txt comes from Wikipedia .
Text is available under the Creative Commons Attribution-ShareAlike License 4.0

# Roadmap

- implement cross-entropy loss
- optimize code to reduce calls to malloc() and free() -> avoid transpose and instead implement a matmul with lhs transpose semantics
- shuffle examples in each epoch
- bpe tokenizer
- add tape like in pytorch
- rename matrix to tensor
- implement transformer
- gelu


# Links

- [Training Hidden Units: The Generalized Delta Rule](https://web.stanford.edu/group/pdplab/originalpdphandbook/Chapter%205.pdf)
- [LINEAR](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- [Introducnn.Linear in PyTorch: Clearly Explainedtion](https://docs.kanaries.net/topics/Python/nn-linear)
- [Word Embeddings: Encoding Lexical Semantics](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html)
- [A Simple Introduction to Broadcasting](https://medium.com/@hunter-j-phillips/)a-simple-introduction-to-broadcasting-db8e581368b3
- [Fast Multidimensional Matrix Multiplication on CPU from Scratch](https://siboehm.com/articles/22/Fast-MMM-on-CPU)
- [Mega man](https://en.wikipedia.org/wiki/Mega_Man)
- [Prof. Geoffrey Hinton - "Will digital intelligence replace biological intelligence?" Romanes Lecture](https://www.youtube.com/watch?v=N1TEjTeQeg0)
- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [GAUSSIAN ERROR LINEAR UNITS (GELUS)](https://arxiv.org/pdf/1606.08415.pdf)
