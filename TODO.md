- colored_mosaic_puzzles: generate examples rotations

== Performance ==

- simplify train.rs to have at most 1 call to infer, loss, compute_gradient, optimize() per example per epoch.
- print total_loss of each epoch

== Story: performance ==

- investigate calls (17% CPU cycles) to pthread_mutex_unlock@@GLIBC_2.2.5 and pthread_rwlock_unlock@@GLIBC_2.34
- debug performance with NVIDIA Nsight Systems

== Story: RMSNorm ==

- implement RMSNorm : https://arxiv.org/pdf/1910.07467

== Backlog ==

- use 4 layers in transformer model (find bug in CPU memory usage that causes a hang)
- rename to ResidualSumOfSquares


- rematerialize dropout mask to save GPU VRAM memory

- remove forward method in tensor
- don't use ClipNorm in AdamW

- set maximum_incorrect_predicted_next_tokens to 0 in transformer dataset
- increase examples in transformer test from 30 to 100

== correct mini-batch implementation ==

- have one unified set for instructions, streams, scheduler instead of four (inference, loss, gradient, optimization) using instruction range (begin..end)
- use batch aggregated loss to compute gradient
- impement mini-batch in the model input tensor shape
- implement mini batch using broadcasting in the operators

== Story: async copy ==

- use result::memcpy_htod_async(*dst.device_ptr_mut(), src, self.stream) to do set_value
- implement htod_into_on_stream in cudarc
- implement set_value_with_stream

== Story: Mega-man transformer ==

- re-add method zero_grad

== Story: use device pointer mode ==

- use device pointer mode for Gemm's alpha and beta (maybe this is the cause of pthread_rwlock_unlock)

== Story: gradient accumulation ==

- honour requires_grad() when updating gradients
- Make sure that backward instruction add on top of existing gradients (no overwrite)

- Implement Transformer idea for colored mosaic puzzles (left-to-right residual connections)

== Clean-up ==

- simplify code that push gradient_instruction instructions (too much re-mapping of inputs to outputs)

- remove all calls to set_values
- implement Transpose with CUDA

- refactor gelu, sigmoid, gelu_derivative in cpu module because they duplicate code
- move ./src/devices/cuda/tests.rs tests that are not related to cuda to ./src/devices/tests.rs
- move ./src/devices/cpu/tests.rs tests that are not related to cpu to ./src/devices/tests.rs

== Story: AMD ROCm with HIP ==

- Add support AMD GPUs (ROCm/rocBLAS) -> https://docs.rs/simt_rocblas_sys/latest/simt_rocblas_sys/struct.rocblas.html

== Backlog ==

---------------

- implement ArgMax operator https://onnx.ai/onnx/operators/onnx__ArgMax.html
- rename RowMax to ArgMax (https://onnx.ai/onnx/operators/onnx__ArgMax.html)
- add code that discard useless instructions, for example when a operand write is never read before the next write

---------------

- add Tensor categories
- use Category::Constant to determine constants
- use Category::Parameter to determine parameters

---------------

- device interface uses <T>
- Implement code with f16

---------------------

== Other things ==

- investigate calls to Device::tensor_f32

== Tensor clean-up ==

- device.tensor should take size &[usize] instead of rows, cols

== Refactoring ==

- merge the many load_examples / generate_examples functions

== Fixes ==

- remove random calls to unwrap()
- return ErrNoGradient if output tensor has no gradient

- image transformer

== Datasets ==

- CIFAR-10
- MNIST

== Import / Export ==

- serialize and deserialize model to ONNX format

== Positional Encoding ==

- implement positional encoding

== Devices ==

- Add support for Jim Keller's https://tenstorrent.com/cards/
- Add support for Google TPU
- Add support for Apple Metal
- Add support for Intel Arc
