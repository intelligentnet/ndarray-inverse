# Inverse and determinant trait for ndarray Array2

When creating some kalman filter code there was a need for Matrix inversion...

Installing external code to calculate inverses was clunky and painful. Hence
this little trait. It's 'reasonably' quick for small Matrices, a dedicated
library, even on a CPU is much faster for bigger Matrices, as you would expect.
If you use relatively small matrices, on multiple operating systems,
then then that is the use case here.
With huge matrices for Machine Learning, you should/would be using
iterative/approximate methods anyway, such as gradient decent. 

If your priority is ease of installation, pure Rust and portability then this
code might be useful.

Simply reference with:
```
use ndarray_inverse::Inverse;
```

Add a cargo dependency too:
```
ndarray-inverse = "*"
```

Available methods on Array2 matrices are m.inv() and m.det(). Obviously they must be square matrices.

There is also m.inv_diag() which will give the inverse of a diagonal matrix, which is much faster but any off diagonal non-zero values are ignored. This is order N rather than order N^3 and is sometimes useful and might be considered a 'normalisation' for some applications.

Version 1.7 added m.cholesky() which is useful for the Unscented Kalman Filter.

Version 1.8 added m.lu() decomposition and associated inverse (m.lu_inv())
which is much faster for larger matrices by ~10x. If it is not appropriate it
falls back to the standard method seamlessly. This makes single CPU matrix
inverse calculation speed competitive with external library versions in most
cases.
