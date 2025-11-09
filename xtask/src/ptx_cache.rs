use std::env;
use std::fs;
use std::io;
use std::path::PathBuf;

/// Get the PTX cache directory
fn get_cache_dir() -> PathBuf {
    if let Ok(dir) = env::var("RUST_CUDA_PTX_CACHE") {
        PathBuf::from(dir)
    } else {
        let home = env::var("HOME")
            .or_else(|_| env::var("USERPROFILE"))
            .unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home).join(".rust-cuda-cache")
    }
}

/// Statistics about cache usage
#[derive(Debug, Clone, Default)]
struct CacheStats {
    entry_count: usize,
    total_bytes: u64,
}

impl CacheStats {
    fn size_mb(&self) -> f64 {
        self.total_bytes as f64 / (1024.0 * 1024.0)
    }
}

/// Get statistics about the PTX cache
fn get_stats() -> io::Result<CacheStats> {
    let cache_dir = get_cache_dir();

    if !cache_dir.exists() {
        return Ok(CacheStats::default());
    }

    let mut stats = CacheStats::default();

    for entry in fs::read_dir(&cache_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("ptx") {
            if let Ok(metadata) = entry.metadata() {
                stats.entry_count += 1;
                stats.total_bytes += metadata.len();
            }
        }
    }

    Ok(stats)
}

/// Clear the PTX cache
fn clear_cache() -> io::Result<()> {
    let cache_dir = get_cache_dir();

    if !cache_dir.exists() {
        println!("Cache directory does not exist: {}", cache_dir.display());
        return Ok(());
    }

    let mut removed = 0;
    let mut failed = 0;

    for entry in fs::read_dir(&cache_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) == Some("ptx") {
            match fs::remove_file(&path) {
                Ok(_) => removed += 1,
                Err(e) => {
                    eprintln!("Failed to remove {}: {}", path.display(), e);
                    failed += 1;
                }
            }
        }
    }

    println!(
        "Cache cleared: {} files removed, {} failures",
        removed, failed
    );

    Ok(())
}

/// Show PTX cache statistics
pub fn stats() -> io::Result<()> {
    let cache_dir = get_cache_dir();
    let stats = get_stats()?;

    println!("PTX Cache Statistics");
    println!("====================");
    println!("Cache directory: {}", cache_dir.display());
    println!("Cache entries:   {}", stats.entry_count);
    println!("Total size:      {:.2} MB", stats.size_mb());

    if stats.entry_count > 0 {
        let avg_size = stats.total_bytes as f64 / stats.entry_count as f64 / 1024.0;
        println!("Average size:    {:.2} KB", avg_size);
    }

    println!();
    println!("Environment variables:");
    println!(
        "  RUST_CUDA_PTX_CACHE_DISABLE = {}",
        env::var("RUST_CUDA_PTX_CACHE_DISABLE").unwrap_or_else(|_| "(not set)".to_string())
    );
    println!(
        "  RUST_CUDA_PTX_CACHE         = {}",
        env::var("RUST_CUDA_PTX_CACHE").unwrap_or_else(|_| "(not set, using default)".to_string())
    );

    Ok(())
}

/// Clear the PTX cache
pub fn clear() -> io::Result<()> {
    clear_cache()
}

/// Enable the PTX cache by removing the disable flag
pub fn enable() {
    env::remove_var("RUST_CUDA_PTX_CACHE_DISABLE");
    println!("PTX cache enabled (RUST_CUDA_PTX_CACHE_DISABLE removed)");
    println!("Note: This only affects the current process.");
    println!("To permanently enable, remove RUST_CUDA_PTX_CACHE_DISABLE from your environment.");
}

/// Disable the PTX cache by setting the disable flag
pub fn disable() {
    env::set_var("RUST_CUDA_PTX_CACHE_DISABLE", "1");
    println!("PTX cache disabled (RUST_CUDA_PTX_CACHE_DISABLE=1)");
    println!("Note: This only affects the current process.");
    println!("To permanently disable, set RUST_CUDA_PTX_CACHE_DISABLE=1 in your environment.");
}

/// Print usage information for cache commands
pub fn usage() {
    println!("PTX Cache Management Commands:");
    println!();
    println!("  cargo xtask cache stats    - Show cache statistics");
    println!("  cargo xtask cache clear    - Clear all cached PTX files");
    println!("  cargo xtask cache enable   - Enable PTX caching (for this process)");
    println!("  cargo xtask cache disable  - Disable PTX caching (for this process)");
    println!();
    println!("Environment Variables:");
    println!("  RUST_CUDA_PTX_CACHE_DISABLE=1  - Disable PTX caching globally");
    println!("  RUST_CUDA_PTX_CACHE=<path>     - Set custom cache directory");
    println!();
    println!("Default cache location: ~/.rust-cuda-cache/");
}
