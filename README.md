# Run the program

cargo build --release
time RUST_BACKTRACE=1 cargo run &> log

# Mega_man

Mega_man.txt comes from https://en.wikipedia.org/wiki/Mega_Man .
Text is available under the Creative Commons Attribution-ShareAlike License 4.0


# Roadmap

- embeddings
- move all links in README
- rename matrix to tensor
- configure network according to dataset
- load megaman
- implement transformer
- gelu

# Links

- https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
