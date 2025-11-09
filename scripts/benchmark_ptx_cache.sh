#!/usr/bin/env bash
set -e

# PTX Cache Benchmark Script
#
# This script measures the performance improvement from PTX caching
# by building a CUDA project multiple times and comparing build times.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default test project
TEST_PROJECT="${1:-examples/cuda/vecadd}"

echo -e "${BLUE}PTX Cache Benchmark${NC}"
echo -e "${BLUE}===================${NC}"
echo ""
echo "Test project: $TEST_PROJECT"
echo "Cache directory: ${RUST_CUDA_PTX_CACHE:-~/.rust-cuda-cache}"
echo ""

# Function to build and measure time
build_and_time() {
    local description="$1"
    local clean="${2:-no}"

    if [ "$clean" = "yes" ]; then
        cargo clean -p "$(basename "$TEST_PROJECT")" 2>/dev/null || true
        cargo clean -p "$(basename "$TEST_PROJECT")-kernels" 2>/dev/null || true
    fi

    echo -e "${YELLOW}Building: $description${NC}"

    # Use GNU time if available, otherwise use bash's time
    if command -v /usr/bin/time &> /dev/null; then
        /usr/bin/time -f "  Time: %E (wall clock)" cargo build --release 2>&1 | grep -E "(Compiling|Finished|Time:)"
    else
        { time cargo build --release 2>&1 | grep -E "(Compiling|Finished)"; } 2>&1 | tail -5
    fi
    echo ""
}

cd "$PROJECT_DIR/$TEST_PROJECT"

# Step 1: Clear cache and do initial build
echo -e "${GREEN}Step 1: Initial build (cache population)${NC}"
cargo xtask cache clear 2>/dev/null || echo "Cache already empty"
export RUST_CUDA_PTX_CACHE_DISABLE=0
unset RUST_CUDA_PTX_CACHE_DISABLE
build_and_time "Initial build - populating cache" "yes"

# Step 2: Rebuild without changes (should hit cache)
echo -e "${GREEN}Step 2: Rebuild without changes (should hit cache)${NC}"
build_and_time "Rebuild - cache hit expected" "yes"

# Step 3: Rebuild with cache disabled
echo -e "${GREEN}Step 3: Rebuild with cache disabled${NC}"
export RUST_CUDA_PTX_CACHE_DISABLE=1
build_and_time "Rebuild - cache disabled" "yes"
unset RUST_CUDA_PTX_CACHE_DISABLE

# Step 4: Show cache stats
echo -e "${GREEN}Cache Statistics:${NC}"
cd "$PROJECT_DIR"
cargo xtask cache stats

echo ""
echo -e "${BLUE}Benchmark Complete!${NC}"
echo ""
echo "Expected results:"
echo "  - Step 1: Normal build time (populates cache)"
echo "  - Step 2: ~50-90% faster (cache hit)"
echo "  - Step 3: Similar to Step 1 (cache disabled)"
echo ""
echo "To clear the cache: cargo xtask cache clear"
