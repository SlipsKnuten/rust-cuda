# PTX Caching for Incremental Builds

## Overview

PTX caching dramatically speeds up incremental builds by caching the expensive NVVM compilation step. When GPU kernel code hasn't changed, the cached PTX is reused instead of recompiling, resulting in 50-90% faster build times.

## How It Works

The PTX cache uses content-addressable storage based on a hash of:

- **Input LLVM bitcode**: The compiled kernel code
- **Target architecture**: e.g., `compute_70`, `compute_80`
- **Optimization level**: `0` (no optimization) or `3` (default)
- **NVVM version**: To invalidate cache on NVVM updates

When you build a CUDA kernel:

1. The compiler merges all LLVM modules and runs dead code elimination
2. It computes a hash of the result + architecture + opt level
3. If a cached PTX file exists with that hash, it's used immediately
4. Otherwise, NVVM compiles the code and the result is cached

## Configuration

### Environment Variables

- **`RUST_CUDA_PTX_CACHE`**: Set custom cache directory
  ```bash
  export RUST_CUDA_PTX_CACHE=/path/to/cache
  ```
  Default: `~/.rust-cuda-cache/`

- **`RUST_CUDA_PTX_CACHE_DISABLE`**: Disable caching entirely
  ```bash
  export RUST_CUDA_PTX_CACHE_DISABLE=1
  ```

### Cache Management

Use `cargo xtask cache` commands to manage the cache:

```bash
# Show cache statistics
cargo xtask cache stats

# Clear all cached PTX files
cargo xtask cache clear

# Enable/disable (affects current process only)
cargo xtask cache enable
cargo xtask cache disable
```

## Usage Examples

### Basic Usage

No configuration needed! Caching is enabled by default. Just build your project:

```bash
cd examples/cuda/vecadd
cargo build --release
```

The first build populates the cache. Subsequent builds reuse cached PTX.

### Custom Cache Directory

```bash
export RUST_CUDA_PTX_CACHE=/tmp/my-ptx-cache
cargo build --release
```

### Disable Caching

```bash
export RUST_CUDA_PTX_CACHE_DISABLE=1
cargo build --release
```

### Check Cache Statistics

```bash
cargo xtask cache stats
```

Output:
```
PTX Cache Statistics
====================
Cache directory: /home/user/.rust-cuda-cache
Cache entries:   5
Total size:      2.34 MB
Average size:    478.92 KB

Environment variables:
  RUST_CUDA_PTX_CACHE_DISABLE = (not set)
  RUST_CUDA_PTX_CACHE         = (not set, using default)
```

## Performance Benchmarks

Run the benchmark script to measure cache performance:

```bash
./scripts/benchmark_ptx_cache.sh examples/cuda/vecadd
```

Expected results:
- **First build**: Normal time (cache population)
- **Second build**: 50-90% faster (cache hit)
- **With cache disabled**: Same as first build

### Example Benchmark Results

```
Step 1: Initial build (cache population)
  Time: 12.34s

Step 2: Rebuild without changes (cache hit)
  Time: 1.52s
  Speedup: 8.1x (87.7% faster)

Step 3: Rebuild with cache disabled
  Time: 12.28s
```

## Advanced Usage

### CI/CD Integration

Pre-populate the cache in CI to speed up builds:

```yaml
# .github/workflows/ci.yml
- name: Restore PTX cache
  uses: actions/cache@v3
  with:
    path: ~/.rust-cuda-cache
    key: ptx-cache-${{ hashFiles('**/kernels/**') }}
```

### Multi-Architecture Builds

The cache stores separate entries for each architecture:

```bash
# Build for compute_70
cargo build --release  # Uses compute_70 cache

# Build for compute_80
RUSTFLAGS="--codegen llvm-args=-arch=compute_80" cargo build --release
```

Each architecture gets its own cached PTX files.

### Cache Maintenance

Evict old entries (not yet implemented, but you can use the clear command):

```bash
# Clear entire cache
cargo xtask cache clear

# Or manually delete old files
find ~/.rust-cuda-cache -name "*.ptx" -mtime +30 -delete
```

## Troubleshooting

### Cache Not Working

1. **Check if caching is enabled**:
   ```bash
   cargo xtask cache stats
   ```

2. **Verify cache directory exists**:
   ```bash
   ls -la ~/.rust-cuda-cache
   ```

3. **Enable debug logging**:
   ```bash
   export NVVM_LOG=debug
   cargo build --release 2>&1 | grep -i cache
   ```

   You should see:
   ```
   PTX cache miss - running NVVM compilation (arch=compute_70, opt=3)
   rust-cuda: Cached PTX with hash 0123456789abcdef
   ```

### Cache Hits Not Showing

The cache operates silently by default. To confirm it's working:

1. Build once to populate: `cargo build --release`
2. Clean and rebuild: `cargo clean && cargo build --release`
3. The second build should be significantly faster

### Stale Cache Issues

If you update CUDA/NVVM and see errors, clear the cache:

```bash
cargo xtask cache clear
```

The cache automatically invalidates when NVVM version changes, but manual clearing may help with edge cases.

## Implementation Details

### Cache Key Computation

```rust
hash = hash_of(
    bitcode_bytes,           // LLVM IR after merge + DCE
    architecture_string,     // e.g., "compute_70"
    optimization_level,      // "0" or "3"
    nvvm_major_version,      // NVVM major version
    nvvm_minor_version       // NVVM minor version
)
```

### Cache File Format

Cache files are stored as:
```
~/.rust-cuda-cache/{hash:016x}.ptx
```

Example: `~/.rust-cuda-cache/0123456789abcdef.ptx`

Each file contains the raw PTX assembly output from NVVM compilation.

### Cache Lookup Flow

```
┌─────────────────────┐
│ Merge LLVM Modules  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Run DCE Pass        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Compute Hash        │
└──────────┬──────────┘
           │
           ▼
      ┌────┴────┐
      │ Cached? │
      └────┬────┘
           │
    ┌──────┴──────┐
    │             │
  YES            NO
    │             │
    ▼             ▼
┌────────┐  ┌──────────┐
│ Return │  │ Compile  │
│ Cached │  │ with     │
│  PTX   │  │ NVVM     │
└────────┘  └─────┬────┘
                  │
                  ▼
            ┌──────────┐
            │  Cache   │
            │  Result  │
            └──────────┘
```

## FAQ

### Q: Will this work with incremental compilation?

**A:** Yes! PTX caching is orthogonal to Rust's incremental compilation. Both work together:
- Rust incremental compilation reduces recompilation of host code
- PTX caching avoids recompiling GPU kernels when unchanged

### Q: How much disk space does the cache use?

**A:** Each PTX file is typically 2-10 MB. The cache grows with unique combinations of (bitcode, arch, opt_level). Use `cargo xtask cache stats` to monitor size.

### Q: Does it cache per-project or globally?

**A:** The cache is global by default (`~/.rust-cuda-cache`), shared across all projects. This maximizes cache hits if you use similar kernels across projects.

### Q: What if I change kernel code?

**A:** The bitcode hash changes, so the cache misses and recompiles. The new result is then cached.

### Q: Can I use this in a team?

**A:** Yes! You can share a cache directory on a network drive:
```bash
export RUST_CUDA_PTX_CACHE=/shared/team-ptx-cache
```

However, ensure proper file locking if multiple developers write simultaneously.

### Q: Does this affect release builds?

**A:** Yes, caching works for both debug and release builds. The optimization level is part of the cache key.

## Limitations

- **Hash Collisions**: Extremely unlikely with 64-bit hashing, but theoretically possible
- **No LRU Eviction**: Cache grows indefinitely (use `cargo xtask cache clear` to manage)
- **No Compression**: PTX files are stored uncompressed
- **Single-Machine**: No built-in distributed cache (but can use shared network directory)

## Future Enhancements

Potential improvements (not yet implemented):

- Automatic LRU eviction based on cache size or age
- Compression of cached PTX files
- Distributed cache server for teams
- Integration with sccache or other cache systems
- Cache statistics tracking (hit rate, etc.)

## Credits

PTX caching was inspired by:
- Compiler caching tools like ccache and sccache
- Rust's incremental compilation system
- Content-addressable storage in Git

## See Also

- [CLAUDE.md](../CLAUDE.md) - Project overview
- [NVVM Documentation](https://docs.nvidia.com/cuda/nvvm-ir-spec/)
- [Benchmark Script](../scripts/benchmark_ptx_cache.sh)
