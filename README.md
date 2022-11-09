# Inverse and determinant trait for ndarray Array2

When creating some kalman filter code there was a need for Matrix inversion...

Installing external code to calculate inverses was clunky and painful. Hence
this little trait. It's 'reasonably' quick for small Matrices, a dedicated
library, even on a CPU is ~8x faster for bigger Matrices, as you would expect.
Relatively small matrices / infrequent need for inverses are the use case here.
With huge matrices for Machine Learning, you would be using
iterative/approximate mathods anyway, such as gradient decent. 

If your priority is ease of installation, pure Rust and portability then this
code might be useful.

Simple reference with:
```
use ndarray_inverse::Inverse;
```

Add a cargo dependency too:
```
ndarray-inverse = "*"
```
