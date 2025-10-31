use cuda_std::prelude::*;

#[kernel] //säg att det ska kompileras till ptx kod som kör på gpu
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]

pub unsafe fn matrix_multiplication (a: *const f32, b: *const f32, c: *mut f32, m: usize, n: usize, k: usize){
    let thread_position = thread::index_2d();
    let row = thread_position.x as usize;
    let column = thread_position.y as usize;

    //antalet rader och kolumner i c matrisen, 
    if row >= m || column >= n {
        return;
    }
}