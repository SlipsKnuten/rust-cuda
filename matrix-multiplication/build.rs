// TODO: Implement build script to compile GPU kernels
use std::env;
use std::path;

use cuda_builder::CudaBuilder;

fn main(){
    let output_path = path::PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_path = path::PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    CudaBuilder::new(manifest_path.join("kernels"))
        .copy_to(output_path.join("kernels.ptx"))
        .build()
        .unwrap();

}