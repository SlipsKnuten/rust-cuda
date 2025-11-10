//! PTX caching system for incremental builds
//!
//! This module implements content-addressable caching of compiled PTX code to dramatically
//! speed up incremental builds by avoiding expensive NVVM compilation when the input
//! bitcode, architecture, and optimization level haven't changed.
//!
//! ## Configuration
//!
//! - `RUST_CUDA_PTX_CACHE`: Custom cache directory (default: `~/.rust-cuda-cache`)
//! - `RUST_CUDA_PTX_CACHE_DISABLE=1`: Disable caching entirely
//!
//! ## Cache Strategy
//!
//! The cache key is a 64-bit hash of:
//! - Input LLVM bitcode
//! - Target architecture (e.g., "compute_70")
//! - Optimization level ("0" or "3")
//! - NVVM version (to invalidate cache on NVVM updates)
//!
//! Cache files are stored as `{hash:016x}.ptx` in the cache directory.

use std::collections::hash_map::DefaultHasher;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io;
use std::path::{Path, PathBuf};
use tracing::{debug, warn};

/// Content-addressable PTX cache
pub struct PtxCache {
    cache_dir: PathBuf,
    enabled: bool,
}

/// Statistics about cache usage
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub entry_count: usize,
    pub total_bytes: u64,
}

impl CacheStats {
    pub fn size_mb(&self) -> f64 {
        self.total_bytes as f64 / (1024.0 * 1024.0)
    }
}

impl PtxCache {
    /// Create a new PTX cache instance
    ///
    /// # Errors
    ///
    /// Returns an error if the cache directory cannot be created
    pub fn new() -> io::Result<Self> {
        let enabled = env::var("RUST_CUDA_PTX_CACHE_DISABLE")
            .map(|v| v != "1")
            .unwrap_or(true);

        let cache_dir = if let Ok(dir) = env::var("RUST_CUDA_PTX_CACHE") {
            PathBuf::from(dir)
        } else {
            let home = env::var("HOME")
                .or_else(|_| env::var("USERPROFILE"))
                .unwrap_or_else(|_| ".".to_string());
            PathBuf::from(home).join(".rust-cuda-cache")
        };

        if enabled {
            fs::create_dir_all(&cache_dir)?;
            debug!("PTX cache initialized at: {}", cache_dir.display());
        } else {
            debug!("PTX cache disabled via RUST_CUDA_PTX_CACHE_DISABLE");
        }

        Ok(Self { cache_dir, enabled })
    }

    /// Compute a 64-bit hash of the cache key components
    fn compute_hash(&self, bitcode: &[u8], arch: &str, opt_level: &str) -> u64 {
        let mut hasher = DefaultHasher::new();

        // Hash the bitcode content
        bitcode.hash(&mut hasher);

        // Hash the architecture
        arch.hash(&mut hasher);

        // Hash the optimization level
        opt_level.hash(&mut hasher);

        // Hash the NVVM version to invalidate cache on NVVM updates
        let (major, minor) = nvvm::nvvm_version();
        major.hash(&mut hasher);
        minor.hash(&mut hasher);

        hasher.finish()
    }

    /// Get the cache file path for a given hash
    fn cache_path(&self, hash: u64) -> PathBuf {
        self.cache_dir.join(format!("{:016x}.ptx", hash))
    }

    /// Try to get cached PTX for the given inputs
    ///
    /// Returns `Some(ptx_string)` if a cache hit occurs, `None` otherwise
    pub fn get(&self, bitcode: &[u8], arch: &str, opt_level: &str) -> Option<Vec<u8>> {
        if !self.enabled {
            return None;
        }

        let hash = self.compute_hash(bitcode, arch, opt_level);
        let path = self.cache_path(hash);

        match fs::read(&path) {
            Ok(ptx) => {
                debug!(
                    "rust-cuda: PTX cache hit for hash {:016x} (arch={}, opt={})",
                    hash, arch, opt_level
                );
                Some(ptx)
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                debug!(
                    "rust-cuda: PTX cache miss for hash {:016x} (arch={}, opt={})",
                    hash, arch, opt_level
                );
                None
            }
            Err(e) => {
                warn!(
                    "rust-cuda: Failed to read PTX cache at {}: {}",
                    path.display(),
                    e
                );
                None
            }
        }
    }

    /// Store PTX in the cache
    ///
    /// # Errors
    ///
    /// Returns an error if the cache file cannot be written
    pub fn put(&self, bitcode: &[u8], arch: &str, opt_level: &str, ptx: &[u8]) -> io::Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let hash = self.compute_hash(bitcode, arch, opt_level);
        let path = self.cache_path(hash);

        fs::write(&path, ptx)?;

        debug!(
            "rust-cuda: Cached PTX with hash {:016x} (arch={}, opt={}, size={} bytes)",
            hash,
            arch,
            opt_level,
            ptx.len()
        );

        Ok(())
    }

    /// Clear all entries from the cache
    ///
    /// # Errors
    ///
    /// Returns an error if cache entries cannot be removed
    pub fn clear(&self) -> io::Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let mut removed = 0;
        let mut failed = 0;

        for entry in fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("ptx") {
                match fs::remove_file(&path) {
                    Ok(_) => removed += 1,
                    Err(e) => {
                        warn!("Failed to remove cache file {}: {}", path.display(), e);
                        failed += 1;
                    }
                }
            }
        }

        debug!(
            "rust-cuda: Cleared PTX cache: {} files removed, {} failures",
            removed, failed
        );

        Ok(())
    }

    /// Get statistics about the cache
    ///
    /// # Errors
    ///
    /// Returns an error if the cache directory cannot be read
    pub fn stats(&self) -> io::Result<CacheStats> {
        if !self.enabled {
            return Ok(CacheStats::default());
        }

        let mut stats = CacheStats::default();

        for entry in fs::read_dir(&self.cache_dir)? {
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

    /// Evict cache entries older than the specified number of days
    ///
    /// # Errors
    ///
    /// Returns an error if cache entries cannot be accessed or removed
    pub fn evict_old_entries(&self, max_age_days: u64) -> io::Result<usize> {
        if !self.enabled {
            return Ok(0);
        }

        use std::time::{Duration, SystemTime};

        let cutoff = SystemTime::now() - Duration::from_secs(max_age_days * 86400);
        let mut evicted = 0;

        for entry in fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("ptx") {
                if let Ok(metadata) = entry.metadata() {
                    if let Ok(modified) = metadata.modified() {
                        if modified < cutoff {
                            fs::remove_file(&path)?;
                            evicted += 1;
                        }
                    }
                }
            }
        }

        debug!("rust-cuda: Evicted {} old cache entries", evicted);
        Ok(evicted)
    }

    /// Get the cache directory path
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Check if caching is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn temp_cache() -> io::Result<PtxCache> {
        let temp_dir = env::temp_dir().join(format!("rust-cuda-test-{}", std::process::id()));
        unsafe {
            env::set_var("RUST_CUDA_PTX_CACHE", &temp_dir);
            env::remove_var("RUST_CUDA_PTX_CACHE_DISABLE");
        }

        let cache = PtxCache::new()?;

        // Clean up any existing test cache
        let _ = cache.clear();

        Ok(cache)
    }

    #[test]
    fn test_cache_roundtrip() -> io::Result<()> {
        let cache = temp_cache()?;

        let bitcode = b"test bitcode content";
        let arch = "compute_70";
        let opt_level = "3";
        let ptx = b"test ptx output";

        // Should be cache miss initially
        assert!(cache.get(bitcode, arch, opt_level).is_none());

        // Store in cache
        cache.put(bitcode, arch, opt_level, ptx)?;

        // Should be cache hit now
        let cached = cache.get(bitcode, arch, opt_level).expect("cache hit");
        assert_eq!(cached, ptx);

        // Cleanup
        cache.clear()?;
        Ok(())
    }

    #[test]
    fn test_cache_key_sensitivity() -> io::Result<()> {
        let cache = temp_cache()?;

        let bitcode1 = b"bitcode v1";
        let bitcode2 = b"bitcode v2";
        let arch = "compute_70";
        let opt_level = "3";
        let ptx = b"ptx output";

        cache.put(bitcode1, arch, opt_level, ptx)?;

        // Different bitcode should miss
        assert!(cache.get(bitcode2, arch, opt_level).is_none());

        // Different arch should miss
        assert!(cache.get(bitcode1, "compute_80", opt_level).is_none());

        // Different opt level should miss
        assert!(cache.get(bitcode1, arch, "0").is_none());

        // Same inputs should hit
        assert!(cache.get(bitcode1, arch, opt_level).is_some());

        // Cleanup
        cache.clear()?;
        Ok(())
    }

    #[test]
    fn test_cache_stats() -> io::Result<()> {
        let cache = temp_cache()?;

        let stats_empty = cache.stats()?;
        assert_eq!(stats_empty.entry_count, 0);

        cache.put(b"bc1", "compute_70", "3", b"ptx1")?;
        cache.put(b"bc2", "compute_80", "3", b"ptx2")?;

        let stats = cache.stats()?;
        assert_eq!(stats.entry_count, 2);
        assert!(stats.total_bytes > 0);

        // Cleanup
        cache.clear()?;
        Ok(())
    }

    #[test]
    fn test_cache_disable() -> io::Result<()> {
        unsafe {
            env::set_var("RUST_CUDA_PTX_CACHE_DISABLE", "1");
        }

        let cache = PtxCache::new()?;
        assert!(!cache.is_enabled());

        let bitcode = b"test";
        let arch = "compute_70";
        let opt_level = "3";
        let ptx = b"ptx";

        // Put should succeed but not actually cache
        cache.put(bitcode, arch, opt_level, ptx)?;

        // Get should always miss
        assert!(cache.get(bitcode, arch, opt_level).is_none());

        unsafe {
            env::remove_var("RUST_CUDA_PTX_CACHE_DISABLE");
        }
        Ok(())
    }

    #[test]
    fn test_cache_clear() -> io::Result<()> {
        let cache = temp_cache()?;

        cache.put(b"bc1", "compute_70", "3", b"ptx1")?;
        cache.put(b"bc2", "compute_80", "3", b"ptx2")?;

        let stats_before = cache.stats()?;
        assert_eq!(stats_before.entry_count, 2);

        cache.clear()?;

        let stats_after = cache.stats()?;
        assert_eq!(stats_after.entry_count, 0);

        Ok(())
    }
}
