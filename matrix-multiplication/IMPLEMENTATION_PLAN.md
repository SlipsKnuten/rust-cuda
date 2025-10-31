# Matrix Multiplication Implementation Plan

This document provides a progressive, step-by-step plan for implementing matrix multiplication (GEMM: General Matrix Multiply) using Rust CUDA. You'll start with a simple naive implementation and progressively add optimizations.

**Formula:** `C = A × B` where A is `m×k`, B is `k×n`, and C is `m×n`

---

## Phase 0: Project Setup

### Goal
Set up the build system and project structure to compile GPU kernels and run host code.

### Implementation Steps

1. **Fix Cargo.toml edition**
   - [x] Change `edition = "2024"` to `edition = "2021"` in `Cargo.toml`
   - [x]
2. **Implement build.rs**
   - [x] Import `cuda_builder::CudaBuilder`
   - [x] Get the output directory using `std::env::var("OUT_DIR")`
   - [x] Get the manifest directory using `std::env::var("CARGO_MANIFEST_DIR")`
   - [x] Use `CudaBuilder::new(manifest_dir.join("kernels"))` to specify the kernel crate location
   - [x] Call `.copy_to(out_path.join("kernels.ptx"))` to output the compiled PTX
   - [x] Call `.build()` to trigger compilation

3. **Set up kernel crate dependencies**
   - [x] In `kernels/Cargo.toml`, verify `cuda_std = { workspace = true }` dependency exists
   - [x] Ensure `[lib]` section has `crate-type = ["cdylib", "rlib"]`

4. **Test the build system**
   - [x] Run `cargo build -p matrix-multiplication`
   - [x] Verify that `kernels.ptx` is generated in the target directory
   - [x] Check for any build errors

### Key Concepts
- **PTX (Parallel Thread Execution)**: Assembly-like intermediate representation for NVIDIA GPUs
- **cuda_builder**: Orchestrates the compilation of Rust GPU code to PTX
- **Build script**: Runs before main compilation to generate necessary artifacts

### Reference Files
- `examples/cuda/vecadd/build.rs` - Simple build.rs template
- `examples/cuda/gemm/build.rs` - More complete example

---

## Phase 1: Naive Kernel Implementation

### Goal
Implement a simple matrix multiplication kernel where each thread computes exactly one element of the output matrix C.

### Background
In the naive approach:
- Each thread calculates one `C[i,j]` element
- The thread performs a dot product: `C[i,j] = sum(A[i,k] * B[k,j])` for all k
- Simple to understand but not memory-efficient (no data reuse between threads)

### Implementation Steps

1. **Understand the thread indexing**
   - [x] Each thread needs to determine its output position (row `i`, column `j`)
   - [x] Use `thread::index_2d()` to get the thread's 2D coordinates
   - [x] Calculate global thread indices based on block and thread IDs

2. **Implement the naive kernel in kernels/src/lib.rs**
   - [x] Add `#[kernel]` attribute to your function
   - [x] Add `#[allow(improper_ctypes_definitions)]` to suppress warnings
   - [x] Make the function `pub unsafe fn`
   - [x] Parameters needed: pointers to A, B, C (`*const f32` for inputs, `*mut f32` for output)
   - [x] Add matrix dimensions: `m: usize`, `n: usize`, `k: usize`

3. **Calculate thread's output position**
   - [ ] Get row index `i` from thread's y-coordinate
   - [ ] Get column index `j` from thread's x-coordinate
   - [ ] Add bounds checking: if `i >= m || j >= n`, return early

4. **Compute the dot product**
   - [ ] Initialize accumulator: `let mut sum = 0.0f32`
   - [ ] Loop over the shared dimension k: `for k_idx in 0..k`
   - [ ] Calculate index into A (row-major): `A[i * k + k_idx]`
   - [ ] Calculate index into B (row-major): `B[k_idx * n + j]`
   - [ ] Accumulate: `sum += a_element * b_element`
   - [ ] Remember to use unsafe pointer dereferencing

5. **Write the result**
   - [ ] Calculate output index: `C[i * n + j]`
   - [ ] Write sum to output using pointer arithmetic

### Key Concepts
- **Thread Hierarchy**: Grid → Blocks → Threads
- **Row-major layout**: Element at (row, col) is at `[row * num_cols + col]`
- **Bounds checking**: Critical to prevent out-of-bounds access
- **cuda_std::thread**: Provides `index()`, `index_2d()`, `index_3d()` functions

### Gotchas
- Don't forget bounds checking - matrix dimensions might not be divisible by block size
- Pointer arithmetic in Rust requires unsafe blocks
- Use `*const` for read-only data, `*mut` for writable output

### Reference Files
- `examples/cuda/gemm/kernels/src/gemm_naive.rs` (lines 1-47)

---

## Phase 2: Host Code - Basic Version

### Goal
Write host code that initializes CUDA, loads the kernel, prepares matrices, launches the kernel, and retrieves results.

### Implementation Steps

1. **Include the PTX at compile time**
   - [ ] Add at top of main.rs: `static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));`

2. **Set up error handling**
   - [ ] Change main signature to `fn main() -> Result<(), Box<dyn std::error::Error>>`
   - [ ] Import necessary types from `cust`

3. **Initialize CUDA context**
   - [ ] Call `cust::quick_init()` to initialize CUDA and create a context
   - [ ] Store the context (even if unused) to keep it alive: `let _ctx = ...`

4. **Load the PTX module**
   - [ ] Use `Module::from_ptx(PTX, &[])` to load your compiled kernel
   - [ ] The empty slice `&[]` is for module options (usually none needed)

5. **Get the kernel function**
   - [ ] Use `module.get_function("your_kernel_name")`
   - [ ] Store as `Function` type for launching

6. **Create test matrices**
   - [ ] Start with small matrices for easy verification (e.g., 4×4)
   - [ ] Create Vec<f32> for matrices A, B, and C
   - [ ] Initialize A and B with known values (e.g., sequential numbers or identity)
   - [ ] Initialize C with zeros

7. **Allocate device memory**
   - [ ] Use `DeviceBuffer::from_slice(&a)` to copy A to GPU
   - [ ] Similarly for B
   - [ ] Use `DeviceBuffer::from_slice(&c)` for C (or `zeroed` if starting empty)

8. **Configure launch parameters**
   - [ ] Choose block dimensions (e.g., 16×16 threads per block)
   - [ ] Calculate grid dimensions: `(m + block_x - 1) / block_x` for each dimension
   - [ ] Create `LaunchConfig` with grid and block sizes

9. **Launch the kernel**
   - [ ] Use the `launch!` macro (it's unsafe)
   - [ ] Pass: function, launch config, and all kernel parameters
   - [ ] Remember to pass raw pointers using `.as_device_ptr()` on DeviceBuffers

10. **Retrieve and verify results**
    - [ ] Create a host Vec to receive results
    - [ ] Use `device_c.copy_to(&mut host_c)` to copy back
    - [ ] Print or verify results against expected output

11. **Add basic verification**
    - [ ] Compute expected result on CPU
    - [ ] Compare with GPU result element-by-element
    - [ ] Print success/failure message

### Key Concepts
- **DeviceBuffer**: RAII wrapper for GPU memory that auto-frees on drop
- **Module**: Container for GPU functions compiled from PTX
- **LaunchConfig**: Specifies how many blocks and threads to launch
- **Synchronization**: `copy_to` automatically synchronizes

### Gotchas
- The `launch!` macro requires unsafe because you're calling GPU code
- Kernel parameter order must exactly match the kernel function signature
- Grid/block dimensions use `(x, y, z)` ordering, but you might think in `(rows, cols)`

### Reference Files
- `examples/cuda/gemm/src/main.rs` (lines 1-150 for basic structure)
- `examples/cuda/vecadd/src/main.rs` - Simpler example

---

## Phase 3: Memory Optimization - Shared Memory Tiling

### Goal
Optimize the kernel by using shared memory to cache tiles of A and B, reducing global memory accesses.

### Background
**Why tiling helps:**
- Global memory is slow (~400-800 cycles latency)
- Shared memory is fast (~20-40 cycles latency)
- Multiple threads in a block reuse the same data
- Example: For a 16×16 output tile, each A element is reused 16 times, each B element is reused 16 times

**Tiling strategy:**
- Divide output matrix C into tiles (e.g., 16×16)
- Each thread block computes one output tile
- Load input tiles from A and B into shared memory
- Compute partial results using shared memory data
- Repeat for all tiles along the k dimension

### Implementation Steps

1. **Define tile size constant**
   - [ ] Choose tile size (typically 16 or 32): `const TILE_SIZE: usize = 16`
   - [ ] This must match your block dimensions

2. **Add shared memory arrays in kernel**
   - [ ] Declare `#[shared]` static arrays for A and B tiles
   - [ ] Type: `static mut A_TILE: [[f32; TILE_SIZE]; TILE_SIZE]`
   - [ ] Similarly for B_TILE

3. **Calculate thread indices**
   - [ ] Get block indices: which tile this block is computing
   - [ ] Get local thread indices within the block
   - [ ] Calculate global row and column for final output

4. **Implement tile loop**
   - [ ] Loop over tiles in the k dimension: `for tile in 0..((k + TILE_SIZE - 1) / TILE_SIZE)`
   - [ ] Each iteration processes one TILE_SIZE slice of k

5. **Collaboratively load tile from A**
   - [ ] Each thread loads one element of A into shared memory
   - [ ] Calculate which element of A this thread should load
   - [ ] Bounds check: ensure indices are valid
   - [ ] Store in `A_TILE[thread_local_row][thread_local_col]`

6. **Collaboratively load tile from B**
   - [ ] Similar to A, each thread loads one element of B
   - [ ] Store in `B_TILE[thread_local_row][thread_local_col]`

7. **Synchronize threads**
   - [ ] Call `cuda_std::thread::sync_threads()` after loading
   - [ ] This ensures all threads in the block have finished loading before anyone starts computing

8. **Compute partial dot product**
   - [ ] Loop over TILE_SIZE elements
   - [ ] Read from shared memory: `A_TILE[local_row][k]` and `B_TILE[k][local_col]`
   - [ ] Accumulate into your sum variable

9. **Synchronize before next tile**
   - [ ] Call `sync_threads()` again before loading the next tile
   - [ ] Prevents threads from overwriting shared memory while others still read

10. **Write final result**
    - [ ] After all tiles, write accumulated sum to global memory
    - [ ] Same as naive version

### Key Concepts
- **Shared memory**: On-chip memory shared by all threads in a block, much faster than global memory
- **Cooperative loading**: All threads work together to load tiles efficiently
- **Thread synchronization**: `sync_threads()` ensures all threads reach the same point
- **Memory coalescing**: Loading consecutive elements improves bandwidth

### Gotchas
- Must synchronize before and after using shared memory
- Shared memory size is limited (typically 48KB per SM)
- Tile size must be known at compile time (Rust doesn't support dynamic shared memory easily)
- Edge cases: matrices not divisible by TILE_SIZE need special handling

### Reference Files
- `examples/cuda/gemm/kernels/src/gemm_tiled.rs` (complete tiled implementation)

---

## Phase 4: Performance Comparison

### Goal
Compare naive vs tiled performance to understand the impact of optimizations.

### Implementation Steps

1. **Add timing infrastructure**
   - [ ] Import `std::time::Instant`
   - [ ] Add warmup runs (run kernel 2-3 times before timing)
   - [ ] Time multiple iterations (10-100) and average

2. **Create Stream for async operations**
   - [ ] Use `Stream::new()` to create a CUDA stream
   - [ ] Launch kernels on the stream
   - [ ] Synchronize with `stream.synchronize()`

3. **Test multiple matrix sizes**
   - [ ] Start small: 128×128
   - [ ] Test powers of 2: 256, 512, 1024, 2048
   - [ ] Also test non-power-of-2 sizes to verify correctness

4. **Calculate and display metrics**
   - [ ] Compute GFLOPS: `2.0 * m * n * k / (time_in_seconds * 1e9)`
   - [ ] The factor of 2 accounts for multiply-add operations
   - [ ] Display time in milliseconds and GFLOPS

5. **Compare implementations**
   - [ ] Run both naive and tiled kernels on same inputs
   - [ ] Show speedup ratio: `naive_time / tiled_time`
   - [ ] Verify both produce identical results

### Key Concepts
- **GFLOPS**: Giga Floating Point Operations Per Second - standard performance metric
- **Warmup**: First kernel launches are slower due to compilation and caching
- **Streams**: Allow asynchronous GPU operations

### Expected Results
- Tiled version should be 5-20× faster than naive for large matrices
- Speedup increases with matrix size
- Peak performance on modern GPUs: 100-1000+ GFLOPS for matrix multiplication

---

## Phase 5: Advanced Optimizations (Optional)

These are stretch goals for after you have working naive and tiled implementations.

### Potential Optimizations

1. **Register Blocking**
   - Each thread computes multiple output elements (e.g., 4×4 sub-tile)
   - Reduces shared memory usage and increases arithmetic intensity
   - More complex indexing logic

2. **Vectorized Memory Access**
   - Load multiple values at once using wider data types
   - Requires careful alignment
   - Can use `float2` or `float4` types

3. **Double Buffering**
   - Use two sets of shared memory tiles
   - Load next tile while computing current tile
   - Hides memory latency with computation

4. **Warp-Level Programming**
   - Use warp shuffle instructions for communication
   - Reduce shared memory usage
   - Requires understanding of warp execution model

5. **Architecture-Specific Tuning**
   - Different tile sizes for different GPU architectures
   - Use `#[cfg]` attributes to compile different versions
   - Test compute_70, compute_80, compute_86, etc.

### Reference
- Check NVIDIA's CUTLASS library for state-of-the-art GEMM implementations
- Read CUDA C Programming Guide sections on shared memory and optimization

---

## Testing & Validation Checklist

### Correctness Tests
- [ ] Test with identity matrix (should return input)
- [ ] Test with zero matrix (should return zeros)
- [ ] Test with small matrices where you can verify by hand (3×3 or 4×4)
- [ ] Test non-square matrices (m ≠ n ≠ k)
- [ ] Test matrices where dimensions aren't multiples of tile size
- [ ] Compare GPU results with CPU implementation (within floating-point tolerance)

### Performance Tests
- [ ] Run on various matrix sizes: 128, 256, 512, 1024, 2048, 4096
- [ ] Profile with NVIDIA Nsight Compute or nvprof
- [ ] Check occupancy: `cuda_std::occupancy` APIs
- [ ] Monitor memory bandwidth utilization

### Edge Cases
- [ ] Very small matrices (smaller than block size)
- [ ] Very large matrices (test memory allocation limits)
- [ ] Rectangular matrices with extreme aspect ratios (1000×10 × 10×1000)

---

## Common Issues & Solutions

### Build Issues
- **"libnvvm not found"**: Install CUDA Toolkit, set `CUDA_PATH` environment variable
- **"rustc version mismatch"**: Ensure you're using `nightly-2025-08-04` (check `rust-toolchain.toml`)
- **PTX file not found**: Check that build.rs ran successfully, look in `target/debug/build/`

### Runtime Issues
- **Kernel launch fails**: Check grid/block dimensions, ensure they're not too large
- **Wrong results**:
  - Verify row-major indexing is consistent
  - Check bounds in kernel
  - Ensure synchronization points are correct
- **Slow performance**:
  - Are you measuring warmup runs? Exclude them
  - Is the matrix size too small? Optimize for larger matrices
  - Check memory access patterns for coalescing

### Debugging Tips
- Print from kernels using `cuda_std::printf!` macro (limited buffer)
- Start with very small matrices (4×4) you can inspect completely
- Use `cuda-gdb` for stepping through kernel code
- Enable `RUST_BACKTRACE=1` for better error messages on host side

---

## Resources

### Documentation
- Rust CUDA Book: Check if there's documentation in the repo
- CUDA C Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- `cuda_std` crate docs: Run `cargo doc --open -p cuda_std` from repo root

### Example Code in This Repo
- `examples/cuda/vecadd`: Simplest complete example
- `examples/cuda/gemm`: Full-featured matrix multiplication with benchmarks
- `examples/cuda/sha2_crates_io`: More complex kernel logic

### External Resources
- CUTLASS: NVIDIA's template library for GEMM
- Papers on GPU GEMM optimization (search "Volkov GEMM" or "MAGMA GEMM")

---

## Next Steps

1. Start with Phase 0 and work through each phase sequentially
2. Verify each phase works before moving to the next
3. Don't skip the naive implementation - it's essential for understanding
4. Keep the GEMM example open for reference, but try to implement yourself first
5. When stuck, compare your code with the examples to spot differences

Good luck with your implementation! Remember: correctness first, then optimize.
