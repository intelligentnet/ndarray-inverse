use ndarray::prelude::*;
use ndarray_inverse::Inverse;
//use ndarray_linalg::solve::Inverse;
use std::time::Instant;

fn main() {
    let a: Array2<f64> = array![
        [1.0, 1.0, 3.0, 4.0, 9.0, 3.0],
        [10.0, 10.0, 1.0, 2.0, 2.0, 5.0],
        [2.0, 9.0, 6.0, 10.0, 10.0, 9.0],
        [10.0, 9.0, 9.0, 7.0, 3.0, 6.0],
        [7.0, 6.0, 6.0, 2.0, 9.0, 5.0],
        [3.0, 8.0, 1.0, 4.0, 1.0, 5.0]
    ];
    /*
    let a: Array2<f64> = array![
        [10.,  1.,  7.,  1.,  5.],
       [ 2.,  4.,  8.,  3.,  2.],
       [ 5.,  1.,  2.,  9., 10.],
       [ 6.,  9.,  9.,  7.,  3.],
       [ 1.,  8.,  8., 10.,  5.]];
    */
    let start = Instant::now();

    let mut _q = a.inv();
    for _ in 0..1000000 {
        _q = a.inv();
    }
    println!("Time taken: {:?}", start.elapsed());
}
