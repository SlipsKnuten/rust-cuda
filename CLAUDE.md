# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Rust CUDA Project is an ecosystem for writing GPU code fully in Rust using the CUDA Toolkit. It provides a rustc backend (`rustc_codegen_nvvm`) that compiles Rust to NVVM IR, which is then converted to optimized PTX code for NVIDIA GPUs.

**Important**: This project is in early development. Expect bugs and incomplete features.

## Required Toolchain

- **Rust Nightly**: `nightly-2025-08-04` (specified in `rust-toolchain.toml`)
- **Required Components**: clippy, llvm-tools-preview, rust-src, rustc-dev, rustfmt, rust-analyzer
- **CUDA Toolkit**: Required for NVVM library (libnvvm)
- **Optional**: OptiX SDK for raytracing features (set `OPTIX_ROOT` and `OPTIX_ROOT_DIR` environment variables)

## Build Commands

### Basic Build
```bash
cargo build
```

### Build Specific Crate
```bash
cargo build -p <crate-name>
```

### Build the Codegen Backend
```bash
cargo build -p rustc_codegen_nvvm
```

### Run Examples
```bash
cd examples/cuda/vecadd
cargo run --release
```

## Testing

### Run Compile Tests
```bash
# Using the cargo alias
cargo compiletest

# With specific options
cargo compiletest --bless                    # Update test outputs
cargo compiletest --target-arch compute_80   # Test specific GPU architecture
cargo compiletest FILTER                     # Run tests matching filter
```

The compile test system (`tests/compiletests`) is a UI testing framework that:
- Builds `rustc_codegen_nvvm` if not already built
- Compiles test dependencies (core, cuda_std, etc.) for the target GPU architecture
- Runs tests in `tests/compiletests/ui/` and compares output to `.stderr` files
- Pass `--bless` to automatically update expected outputs after intentional changes

### Using xtask
```bash
cargo xtask extract_llfns <file> <dir>

# PTX Cache Management
cargo xtask cache stats     # Show cache statistics
cargo xtask cache clear     # Clear cached PTX files
cargo xtask cache enable    # Enable caching (current process)
cargo xtask cache disable   # Disable caching (current process)
```

### PTX Caching (Incremental Builds)

The compiler includes a PTX caching system that dramatically speeds up incremental builds (50-90% faster) by avoiding recompilation when GPU kernel code hasn't changed.

**Configuration:**
- `RUST_CUDA_PTX_CACHE`: Custom cache directory (default: `~/.rust-cuda-cache`)
- `RUST_CUDA_PTX_CACHE_DISABLE=1`: Disable caching

**Usage:**
```bash
# Cache is enabled by default
cargo build --release

# Check cache statistics
cargo xtask cache stats

# Clear cache
cargo xtask cache clear

# Benchmark cache performance
./scripts/benchmark_ptx_cache.sh examples/cuda/vecadd
```

See [docs/PTX_CACHING.md](docs/PTX_CACHING.md) for detailed documentation.

## Architecture

### Compilation Pipeline

1. **Host Code (CPU)**: Uses `cust` crate to interact with CUDA Driver API
2. **Device Code (GPU)**: Written using `cuda_std`, compiled via `rustc_codegen_nvvm`
3. **Build Process**: `cuda_builder` orchestrates compilation of GPU kernels in build.rs

The typical structure for a CUDA application:
```
my_cuda_app/
├── build.rs           # Uses cuda_builder::CudaBuilder
├── src/
│   └── main.rs        # Host code using cust
└── kernels/
    ├── Cargo.toml     # GPU crate (no_std, depends on cuda_std)
    └── src/
        └── lib.rs     # GPU kernels marked with #[kernel]
```

### Key Crates

**Compiler Backend:**
- `rustc_codegen_nvvm`: Custom rustc backend targeting NVVM IR, produces PTX code
- `nvvm`: Bindings to libnvvm library
- `ptx_compiler`: PTX compilation utilities
- `cuda_builder`: High-level builder for compiling GPU kernels in build scripts

**GPU-Side (Device Code):**
- `cuda_std`: Standard library for GPU kernels (thread indices, shared memory, warp intrinsics, etc.)
- `cuda_std_macros`: Procedural macros for GPU code (e.g., `#[kernel]`)
- `gpu_rand`: GPU-friendly random number generation

**CPU-Side (Host Code):**
- `cust`: High-level CUDA Driver API wrapper with RAII and Rust error handling
- `cust_core`: Core types shared between host and device
- `cust_raw`: Low-level FFI bindings to CUDA Driver API
- `cust_derive`: Derive macros for `cust`

**Specialized Libraries:**
- `cudnn`: Deep neural network primitives
- `optix`: Hardware raytracing and denoising
- `optix_device`: Device-side OptiX functionality

### Cargo Workspace Structure

The workspace uses `resolver = "2"` and includes:
- `crates/*`: Core libraries
- `examples/cuda/*`: Example applications (vecadd, gemm, path_tracer, sha2_crates_io)
- `examples/optix/*`: OptiX examples
- `tests/compiletests`: Compiler UI tests
- `xtask`: Utility tasks

### Build Configuration

**Important Profile Settings:**
- `rustc_codegen_nvvm` is always built with `opt-level = 3` in dev mode (required for performance)

**Workspace Dependencies:**
- `cuda_std` and `cuda_builder` are workspace dependencies used throughout

### GPU Kernel Compilation

When `cuda_builder::CudaBuilder` runs in build.rs:
1. Invokes rustc with `-Zcodegen-backend=rustc_codegen_nvvm`
2. Passes GPU-specific flags: `-Cno-redzone=yes`, `-Cpanic=abort`, `-Cllvm-args=-arch=<compute_XX>`
3. Builds the kernel crate as `no_std` with `#![feature(abi_ptx)]`
4. Outputs PTX assembly that can be loaded by the CUDA Driver API

The PTX file is typically copied to `OUT_DIR` and then loaded at runtime using `cust::Module::from_ptx()`.

## Common Workflows

### Creating a New CUDA Example

1. Create directories: `examples/cuda/my_example/` and `examples/cuda/my_example/kernels/`
2. Add both to workspace members in root `Cargo.toml`
3. In `kernels/Cargo.toml`: Set up as `no_std` with `cuda_std` dependency
4. In `my_example/build.rs`: Use `CudaBuilder::new("kernels").copy_to(out_path).build()`
5. Write kernels in `kernels/src/lib.rs` with `#[kernel]` attribute
6. Write host code in `my_example/src/main.rs` using `cust`

### Modifying the Codegen Backend

After making changes to `rustc_codegen_nvvm`:
1. Rebuild: `cargo build -p rustc_codegen_nvvm`
2. Test with compile tests: `cargo compiletest --target-arch compute_70`
3. The compiletests will automatically rebuild dependencies as needed

### Working with UI Tests

- Tests are in `tests/compiletests/ui/`
- Each test is a `.rs` file with special comments for directives
- After fixing a bug or changing error messages, run `cargo compiletest --bless`
- Tests can target specific architectures using `// only-compute_XX` or `// ignore-compute_XX`
