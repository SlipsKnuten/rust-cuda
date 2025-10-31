use cust::prelude::*;
use nanorand::{Rng, WyRand};
use std::error::Error;

const NUMBERS_LEN: usize = 100_000;

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

fn main() -> Result<(), Box<dyn Error>> {
    // Generate random test vectors
    let mut wyrand = WyRand::new();
    let mut lhs = vec![2.0f32; NUMBERS_LEN];
    wyrand.fill(&mut lhs);
    let mut rhs = vec![0.0f32; NUMBERS_LEN];
    wyrand.fill(&mut rhs);

    // Initialize CUDA and create context
    let _ctx = cust::quick_init()?;

    // Load the PTX module
    let module = Module::from_ptx(PTX, &[])?;

    // Create a CUDA stream
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Allocate GPU memory and copy input data
    let lhs_gpu = lhs.as_slice().as_dbuf()?;
    let rhs_gpu = rhs.as_slice().as_dbuf()?;

    // Allocate output buffer
    let mut out = vec![0.0f32; NUMBERS_LEN];
    let out_buf = out.as_slice().as_dbuf()?;

    // Get kernel and calculate optimal launch configuration
    let vecadd = module.get_function("vecadd")?;
    let (_, block_size) = vecadd.suggested_launch_configuration(0, 0.into())?;
    let grid_size = (NUMBERS_LEN as u32).div_ceil(block_size);

    println!("using {grid_size} blocks and {block_size} threads per block");

    // Launch the kernel
    unsafe {
        launch!(
            vecadd<<<grid_size, block_size, 0, stream>>>(
                lhs_gpu.as_device_ptr(),
                lhs_gpu.len(),
                rhs_gpu.as_device_ptr(),
                rhs_gpu.len(),
                out_buf.as_device_ptr(),
            )
        )?;
    }

    // Wait for kernel to finish
    stream.synchronize()?;

    // Copy results back from GPU
    out_buf.copy_to(&mut out)?;

    println!("{} + {} = {}", lhs[0], rhs[0], out[0]);

    Ok(())
}
