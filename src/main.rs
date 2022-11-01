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
    let inv = a.inv().expect("Sods Law");

    println!("Inverse: {:?}", inv);

    let start = Instant::now();

    let mut _q = a.inv();
    for _ in 0..1000000 {
        _q = a.inv();
    }
    println!("Time taken: {:?}", start.elapsed());
}
