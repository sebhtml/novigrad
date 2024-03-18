# Run the program

cargo build --release
time RUST_BACKTRACE=1 cargo run &> log

# Mega_man

Mega_man.txt comes from Wikipedia .
Text is available under the Creative Commons Attribution-ShareAlike License 4.0

# Roadmap

- rename matrix to tensor
- implement Linear layer
- move action things to a activation module
- load megaman
- implement transformer
- gelu

# Links

- back-propagation: https://web.stanford.edu/group/pdplab/originalpdphandbook/Chapter%205.pdf
- linear layer: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
- linear layer: https://docs.kanaries.net/topics/Python/nn-linear
- embeddings: https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
- tensor broadcasting: https://medium.com/@hunter-j-phillips/a-simple-introduction-to-broadcasting-db8e581368b3
- matrix multiplication: https://siboehm.com/articles/22/Fast-MMM-on-CPU
- Mega man dataset: https://en.wikipedia.org/wiki/Mega_Man
