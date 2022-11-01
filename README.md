# Inverse and determinant trait for ndarray Array2

When creating some kalman filter code there was a need for Matrix inversion.
Installing external code to calculate inverses was clunky and painful. Hence
this little trait. It's 'reasonably' quick for small Matrices, a dedicated
library, even on a CPU is ~8x faster for bigger Matrices.

If your priority is ease of installation, pure Rust and portability then this might be useful.

Simple reference with:
```
use ndarray_inverse::Inverse;
```

Add a cargo dependency too:
```
ndarray-inverse = "*"
```

