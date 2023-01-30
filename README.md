# Inverse and determinant trait for ndarray

When creating Kalman Filter code there was a need for Matrix inversion...

Installing external code to calculate inverses was clunky and painful. Hence
this little trait. It's quick for smaller Matrices, however a dedicated
library using specialist hardware, is much faster for large Matrices,
as you would expect. That said you should/would be using
iterative/approximate methods anyway, such as gradient decent for
Machine Learning applications. 

If you use relatively small to medium matrices, on multiple operating systems,
without fancy hardware, then that is the use case here.

If your priority is ease of installation, pure Rust and portability then this
code might be useful. The API is trivial and intuitive. You can get on with writing code for your application and not worry about complex components on multiple operating systems.

Simply reference with:
```
use ndarray_inverse::Inverse;
```

Add a cargo dependency too:
```
ndarray-inverse = "*"
```

Available methods on Array2 matrices are m.inv() and m.det(). Obviously they must be square matrices.

Version 1.6 m.inv_diag() added, which will give the inverse of a diagonal matrix, which is much faster but any off diagonal non-zero values are ignored. This is very fast and is sometimes useful and might be considered a 'regularisation' for some applications (reinforcement learning using Kalman filters for example).

Version 1.7 added m.cholesky() which is useful for the Unscented Kalman Filter.

Version 1.8 added m.lu() decomposition (which returns a tuple of (L,U,P) values
and an associated inverse (m.lu_inv()), 
which is much faster for larger matrices by ~10x. The main inverse function is still m.inv() which tries LU decomposition but if it is not appropriate 
falls back to the standard method seamlessly. This makes single CPU matrix
inverse calculation speed competitive with external library versions in most
cases.

Version 1.9 make upper triangular m.inv_ut() and lower triangular m.inv_lt() inverses available to be called externally.
