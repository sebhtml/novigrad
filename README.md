RUSTFLAGS="--emit asm" cargo build --release
time RUST_BACKTRACE=1 cargo run &> log