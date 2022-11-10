use ndarray::prelude::*;
use ndarray::ScalarOperand;
use num_traits::{Float, One, Zero};
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Mul, Neg, Sub};

pub trait Inverse<T: Float> {
    fn det(&self) -> T;
    fn inv(&self) -> Option<Self>
    where
        Self: Sized;
}

impl<T> Inverse<T> for Array2<T>
where
    T: Float + Debug + ScalarOperand + Sum<T>,
{
    fn det(&self) -> T {
        determinant(self)
    }

    fn inv(&self) -> Option<Self> {
        fn submat<'a, T: Float>(res: &'a mut Vec<T>, m: &'a Array2<T>, sr: usize, sc: usize) {
            let s = m.raw_dim();
            //let l = s[0] - 1;
            let mut i: usize = 0;
            (0..s[0]).for_each(|r| (0..s[1]).for_each(|c| {
                if r != sr && c != sc {
                    res[i] = m[(r, c)];
                    i += 1;
                }
            }));
            //(i..l*l).for_each(|i| res[i] = T::zero()); // must fill rest with 0
            // Note: to stop copy we do not return anything. Variable mutated!
        }

        let s = self.raw_dim();
        assert!(s[0] == s[1]);
        let l = s[0];
        let det = determinant(self);
        if !det.is_zero() {
            match l {
                1 => Some(array![[self[(0, 0)].recip()]]),
                2 => Some(array![
                    [self[(1, 1)] / det, -self[(0, 1)] / det],
                    [-self[(1, 0)] / det, self[(0, 0)] / det],
                ]),
                3 => {
                    let m00 = self[(0, 0)];
                    let m01 = self[(0, 1)];
                    let m02 = self[(0, 2)];
                    let m10 = self[(1, 0)];
                    let m11 = self[(1, 1)];
                    let m12 = self[(1, 2)];
                    let m20 = self[(2, 0)];
                    let m21 = self[(2, 1)];
                    let m22 = self[(2, 2)];

                    let x00 = m11 * m22 - m21 * m12;
                    let x01 = m02 * m21 - m01 * m22;
                    let x02 = m01 * m12 - m02 * m11;
                    let x10 = m12 * m20 - m10 * m22;
                    let x11 = m00 * m22 - m02 * m20;
                    let x12 = m10 * m02 - m00 * m12;
                    let x20 = m10 * m21 - m20 * m11;
                    let x21 = m20 * m01 - m00 * m21;
                    let x22 = m00 * m11 - m10 * m01;

                    Some(array![
                        [x00 / det, x01 / det, x02 / det],
                        [x10 / det, x11 / det, x12 / det],
                        [x20 / det, x21 / det, x22 / det]
                    ])
                }
                4 => {
                    let m00 = self[(0, 0)];
                    let m01 = self[(0, 1)];
                    let m02 = self[(0, 2)];
                    let m03 = self[(0, 3)];
                    let m10 = self[(1, 0)];
                    let m11 = self[(1, 1)];
                    let m12 = self[(1, 2)];
                    let m13 = self[(1, 3)];
                    let m20 = self[(2, 0)];
                    let m21 = self[(2, 1)];
                    let m22 = self[(2, 2)];
                    let m23 = self[(2, 3)];
                    let m30 = self[(3, 0)];
                    let m31 = self[(3, 1)];
                    let m32 = self[(3, 2)];
                    let m33 = self[(3, 3)];

                    let x00 = m11 * m22 * m33
                        - m11 * m32 * m23
                        - m12 * m21 * m33
                        + m12 * m31 * m23
                        + m13 * m21 * m32
                        - m13 * m31 * m22;

                    let x10 = -m10 * m22 * m33
                        + m10 * m32 * m23
                        + m12 * m20 * m33
                        - m12 * m30 * m23
                        - m13 * m20 * m32
                        + m13 * m30 * m22;

                    let x20 = m10 * m21 * m33
                        - m10 * m31 * m23
                        - m11 * m20 * m33
                        + m11 * m30 * m23
                        + m13 * m20 * m31
                        - m13 * m30 * m21;

                    let x30 = -m10 * m21 * m32
                        + m10 * m31 * m22
                        + m11 * m20 * m32
                        - m11 * m30 * m22
                        - m12 * m20 * m31
                        + m12 * m30 * m21;

                    let x01 = -m01 * m22 * m33
                        + m01 * m32 * m23
                        + m02 * m21 * m33
                        - m02 * m31 * m23
                        - m03 * m21 * m32
                        + m03 * m31 * m22;

                    let x11 = m00 * m22 * m33
                        - m00 * m32 * m23
                        - m02 * m20 * m33
                        + m02 * m30 * m23
                        + m03 * m20 * m32
                        - m03 * m30 * m22;

                    let x21 = -m00 * m21 * m33
                        + m00 * m31 * m23
                        + m01 * m20 * m33
                        - m01 * m30 * m23
                        - m03 * m20 * m31
                        + m03 * m30 * m21;

                    let x31 = m00 * m21 * m32
                        - m00 * m31 * m22
                        - m01 * m20 * m32
                        + m01 * m30 * m22
                        + m02 * m20 * m31
                        - m02 * m30 * m21;

                    let x02 = m01 * m12 * m33
                        - m01 * m32 * m13
                        - m02 * m11 * m33
                        + m02 * m31 * m13
                        + m03 * m11 * m32
                        - m03 * m31 * m12;

                    let x12 = -m00 * m12 * m33
                        + m00 * m32 * m13
                        + m02 * m10 * m33
                        - m02 * m30 * m13
                        - m03 * m10 * m32
                        + m03 * m30 * m12;

                    let x22 = m00 * m11 * m33
                        - m00 * m31 * m13
                        - m01 * m10 * m33
                        + m01 * m30 * m13
                        + m03 * m10 * m31
                        - m03 * m30 * m11;

                    let x03 = -m01 * m12 * m23
                        + m01 * m22 * m13
                        + m02 * m11 * m23
                        - m02 * m21 * m13
                        - m03 * m11 * m22
                        + m03 * m21 * m12;

                    let x32 = -m00 * m11 * m32
                        + m00 * m31 * m12
                        + m01 * m10 * m32
                        - m01 * m30 * m12
                        - m02 * m10 * m31
                        + m02 * m30 * m11;

                    let x13 = m00 * m12 * m23
                        - m00 * m22 * m13
                        - m02 * m10 * m23
                        + m02 * m20 * m13
                        + m03 * m10 * m22
                        - m03 * m20 * m12;

                    let x23 = -m00 * m11 * m23
                        + m00 * m21 * m13
                        + m01 * m10 * m23
                        - m01 * m20 * m13
                        - m03 * m10 * m21
                        + m03 * m20 * m11;

                    let x33 = m00 * m11 * m22
                        - m00 * m21 * m12
                        - m01 * m10 * m22
                        + m01 * m20 * m12
                        + m02 * m10 * m21
                        - m02 * m20 * m11;

                    Some(array![
                                [x00 / det, x01 / det, x02 / det, x03 / det],
                                [x10 / det, x11 / det, x12 / det, x13 / det],
                                [x20 / det, x21 / det, x22 / det, x23 / det],
                                [x30 / det, x31 / det, x32 / det, x33 / det]
                    ])
                }
                _ => {
                    // Fully expanding any more is too clunky!
                    if !det.is_zero() {
                        let mut cofactors: Array2<T> = Array2::zeros((l, l));
                        let mut res: Vec<T> = vec![T::zero(); (l - 1) * (l - 1)];
                        for i in 0..l {
                            for j in 0..l {
                                submat(&mut res, self, i, j);
                                let sm: Array2<T> = unsafe { Array2::from_shape_vec_unchecked((l - 1, l - 1), res.to_vec()) };
                                let d = determinant(&sm);
                                cofactors[(j, i)] = if ((i + j) % 2) == 0 { d } else { -d };
                            }
                        }
                        Some(cofactors / det)
                    } else {
                        None
                    }
                }
            }
        } else {
            None
        }
    }
}

fn determinant<T>(m: &Array2<T>) -> T
where
    T: Copy + Zero + One + Mul + Sub<Output=T> + Neg<Output=T> + Sum<T>,
{
    let l = m.raw_dim()[0];
    // Fully expanding the first few determinants makes a big
    // difference, especially as the generic algorithm is recursive
    match l {
        1 => m[(0, 0)],
        2 => m[(0, 0)] * m[(1, 1)] - m[(1, 0)] * m[(0, 1)],
        3 => {
            let m00 = m[(0, 0)];
            let m01 = m[(0, 1)];
            let m02 = m[(0, 2)];
            let m10 = m[(1, 0)];
            let m11 = m[(1, 1)];
            let m12 = m[(1, 2)];
            let m20 = m[(2, 0)];
            let m21 = m[(2, 1)];
            let m22 = m[(2, 2)];

            m00 * (m11 * m22 - m12 * m21) - m01
                * (m10 * m22 - m12 * m20) + m02
                * (m10 * m21 - m11 * m20)
        }
        4 => {
            let m00 = m[(0, 0)];
            let m01 = m[(0, 1)];
            let m02 = m[(0, 2)];
            let m03 = m[(0, 3)];
            let m10 = m[(1, 0)];
            let m11 = m[(1, 1)];
            let m12 = m[(1, 2)];
            let m13 = m[(1, 3)];
            let m20 = m[(2, 0)];
            let m21 = m[(2, 1)];
            let m22 = m[(2, 2)];
            let m23 = m[(2, 3)];
            let m30 = m[(3, 0)];
            let m31 = m[(3, 1)];
            let m32 = m[(3, 2)];
            let m33 = m[(3, 3)];

            // Note: Compiler optimisation will remove redundant multiplies
            m00 * m21 * m32 * m13 -
                m00 * m21 * m33 * m12 -
                m00 * m22 * m31 * m13 +
                m00 * m22 * m33 * m11 +
                m00 * m23 * m31 * m12 -
                m00 * m23 * m32 * m11 -
                m21 * m30 * m02 * m13 +
                m21 * m30 * m03 * m12 -
                m21 * m32 * m03 * m10 +
                m21 * m33 * m02 * m10 +
                m22 * m30 * m01 * m13 -
                m22 * m30 * m03 * m11 +
                m22 * m31 * m03 * m10 -
                m22 * m33 * m01 * m10 -
                m23 * m30 * m01 * m12 +
                m23 * m30 * m02 * m11 -
                m23 * m31 * m02 * m10 +
                m23 * m32 * m01 * m10 +
                m31 * m02 * m13 * m20 -
                m31 * m03 * m12 * m20 -
                m32 * m01 * m13 * m20 +
                m32 * m03 * m11 * m20 +
                m33 * m01 * m12 * m20 -
                m33 * m02 * m11 * m20
        }
        5 => {
            let m00 = m[(0, 0)];
            let m01 = m[(0, 1)];
            let m02 = m[(0, 2)];
            let m03 = m[(0, 3)];
            let m04 = m[(0, 4)];
            let m10 = m[(1, 0)];
            let m11 = m[(1, 1)];
            let m12 = m[(1, 2)];
            let m13 = m[(1, 3)];
            let m14 = m[(1, 4)];
            let m20 = m[(2, 0)];
            let m21 = m[(2, 1)];
            let m22 = m[(2, 2)];
            let m23 = m[(2, 3)];
            let m24 = m[(2, 4)];
            let m30 = m[(3, 0)];
            let m31 = m[(3, 1)];
            let m32 = m[(3, 2)];
            let m33 = m[(3, 3)];
            let m34 = m[(3, 4)];
            let m40 = m[(4, 0)];
            let m41 = m[(4, 1)];
            let m42 = m[(4, 2)];
            let m43 = m[(4, 3)];
            let m44 = m[(4, 4)];

            // Note: Compiler optimisation will remove redundant multiplies
            m00 * m11 * m22 * m33 * m44
                - m00 * m11 * m22 * m34 * m43
                - m00 * m11 * m23 * m32 * m44
                + m00 * m11 * m23 * m34 * m42
                + m00 * m11 * m24 * m32 * m43
                - m00 * m11 * m24 * m33 * m42
                - m00 * m12 * m21 * m33 * m44
                + m00 * m12 * m21 * m34 * m43
                + m00 * m12 * m23 * m31 * m44
                - m00 * m12 * m23 * m34 * m41
                - m00 * m12 * m24 * m31 * m43
                + m00 * m12 * m24 * m33 * m41
                + m00 * m13 * m21 * m32 * m44
                - m00 * m13 * m21 * m34 * m42
                - m00 * m13 * m22 * m31 * m44
                + m00 * m13 * m22 * m34 * m41
                + m00 * m13 * m24 * m31 * m42
                - m00 * m13 * m24 * m32 * m41
                - m00 * m14 * m21 * m32 * m43
                + m00 * m14 * m21 * m33 * m42
                + m00 * m14 * m22 * m31 * m43
                - m00 * m14 * m22 * m33 * m41
                - m00 * m14 * m23 * m31 * m42
                + m00 * m14 * m23 * m32 * m41
                - m01 * m10 * m22 * m33 * m44
                + m01 * m10 * m22 * m34 * m43
                + m01 * m10 * m23 * m32 * m44
                - m01 * m10 * m23 * m34 * m42
                - m01 * m10 * m24 * m32 * m43
                + m01 * m10 * m24 * m33 * m42
                + m01 * m12 * m20 * m33 * m44
                - m01 * m12 * m20 * m34 * m43
                - m01 * m12 * m23 * m30 * m44
                + m01 * m12 * m23 * m34 * m40
                + m01 * m12 * m24 * m30 * m43
                - m01 * m12 * m24 * m33 * m40
                - m01 * m13 * m20 * m32 * m44
                + m01 * m13 * m20 * m34 * m42
                + m01 * m13 * m22 * m30 * m44
                - m01 * m13 * m22 * m34 * m40
                - m01 * m13 * m24 * m30 * m42
                + m01 * m13 * m24 * m32 * m40
                + m01 * m14 * m20 * m32 * m43
                - m01 * m14 * m20 * m33 * m42
                - m01 * m14 * m22 * m30 * m43
                + m01 * m14 * m22 * m33 * m40
                + m01 * m14 * m23 * m30 * m42
                - m01 * m14 * m23 * m32 * m40
                + m02 * m10 * m21 * m33 * m44
                - m02 * m10 * m21 * m34 * m43
                - m02 * m10 * m23 * m31 * m44
                + m02 * m10 * m23 * m34 * m41
                + m02 * m10 * m24 * m31 * m43
                - m02 * m10 * m24 * m33 * m41
                - m02 * m11 * m20 * m33 * m44
                + m02 * m11 * m20 * m34 * m43
                + m02 * m11 * m23 * m30 * m44
                - m02 * m11 * m23 * m34 * m40
                - m02 * m11 * m24 * m30 * m43
                + m02 * m11 * m24 * m33 * m40
                + m02 * m13 * m20 * m31 * m44
                - m02 * m13 * m20 * m34 * m41
                - m02 * m13 * m21 * m30 * m44
                + m02 * m13 * m21 * m34 * m40
                + m02 * m13 * m24 * m30 * m41
                - m02 * m13 * m24 * m31 * m40
                - m02 * m14 * m20 * m31 * m43
                + m02 * m14 * m20 * m33 * m41
                + m02 * m14 * m21 * m30 * m43
                - m02 * m14 * m21 * m33 * m40
                - m02 * m14 * m23 * m30 * m41
                + m02 * m14 * m23 * m31 * m40
                - m03 * m10 * m21 * m32 * m44
                + m03 * m10 * m21 * m34 * m42
                + m03 * m10 * m22 * m31 * m44
                - m03 * m10 * m22 * m34 * m41
                - m03 * m10 * m24 * m31 * m42
                + m03 * m10 * m24 * m32 * m41
                + m03 * m11 * m20 * m32 * m44
                - m03 * m11 * m20 * m34 * m42
                - m03 * m11 * m22 * m30 * m44
                + m03 * m11 * m22 * m34 * m40
                + m03 * m11 * m24 * m30 * m42
                - m03 * m11 * m24 * m32 * m40
                - m03 * m12 * m20 * m31 * m44
                + m03 * m12 * m20 * m34 * m41
                + m03 * m12 * m21 * m30 * m44
                - m03 * m12 * m21 * m34 * m40
                - m03 * m12 * m24 * m30 * m41
                + m03 * m12 * m24 * m31 * m40
                + m03 * m14 * m20 * m31 * m42
                - m03 * m14 * m20 * m32 * m41
                - m03 * m14 * m21 * m30 * m42
                + m03 * m14 * m21 * m32 * m40
                + m03 * m14 * m22 * m30 * m41
                - m03 * m14 * m22 * m31 * m40
                + m04 * m10 * m21 * m32 * m43
                - m04 * m10 * m21 * m33 * m42
                - m04 * m10 * m22 * m31 * m43
                + m04 * m10 * m22 * m33 * m41
                + m04 * m10 * m23 * m31 * m42
                - m04 * m10 * m23 * m32 * m41
                - m04 * m11 * m20 * m32 * m43
                + m04 * m11 * m20 * m33 * m42
                + m04 * m11 * m22 * m30 * m43
                - m04 * m11 * m22 * m33 * m40
                - m04 * m11 * m23 * m30 * m42
                + m04 * m11 * m23 * m32 * m40
                + m04 * m12 * m20 * m31 * m43
                - m04 * m12 * m20 * m33 * m41
                - m04 * m12 * m21 * m30 * m43
                + m04 * m12 * m21 * m33 * m40
                + m04 * m12 * m23 * m30 * m41
                - m04 * m12 * m23 * m31 * m40
                - m04 * m13 * m20 * m31 * m42
                + m04 * m13 * m20 * m32 * m41
                + m04 * m13 * m21 * m30 * m42
                - m04 * m13 * m21 * m32 * m40
                - m04 * m13 * m22 * m30 * m41
                + m04 * m13 * m22 * m31 * m40
        }
        6 => {
            // Okay, getting very unweldy but worth it
            let m00 = m[(0, 0)];
            let m01 = m[(0, 1)];
            let m02 = m[(0, 2)];
            let m03 = m[(0, 3)];
            let m04 = m[(0, 4)];
            let m05 = m[(0, 5)];
            let m10 = m[(1, 0)];
            let m11 = m[(1, 1)];
            let m12 = m[(1, 2)];
            let m13 = m[(1, 3)];
            let m14 = m[(1, 4)];
            let m15 = m[(1, 5)];
            let m20 = m[(2, 0)];
            let m21 = m[(2, 1)];
            let m22 = m[(2, 2)];
            let m23 = m[(2, 3)];
            let m24 = m[(2, 4)];
            let m25 = m[(2, 5)];
            let m30 = m[(3, 0)];
            let m31 = m[(3, 1)];
            let m32 = m[(3, 2)];
            let m33 = m[(3, 3)];
            let m34 = m[(3, 4)];
            let m35 = m[(3, 5)];
            let m40 = m[(4, 0)];
            let m41 = m[(4, 1)];
            let m42 = m[(4, 2)];
            let m43 = m[(4, 3)];
            let m44 = m[(4, 4)];
            let m45 = m[(4, 5)];
            let m50 = m[(5, 0)];
            let m51 = m[(5, 1)];
            let m52 = m[(5, 2)];
            let m53 = m[(5, 3)];
            let m54 = m[(5, 4)];
            let m55 = m[(5, 5)];

            // Note: Compiler optimisation will remove redundant multiplies
            m00 * m11 * m22 * m33 * m44 * m55
                - m00 * m11 * m22 * m33 * m45 * m54
                - m00 * m11 * m22 * m34 * m43 * m55
                + m00 * m11 * m22 * m34 * m45 * m53
                + m00 * m11 * m22 * m35 * m43 * m54
                - m00 * m11 * m22 * m35 * m44 * m53
                - m00 * m11 * m23 * m32 * m44 * m55
                + m00 * m11 * m23 * m32 * m45 * m54
                + m00 * m11 * m23 * m34 * m42 * m55
                - m00 * m11 * m23 * m34 * m45 * m52
                - m00 * m11 * m23 * m35 * m42 * m54
                + m00 * m11 * m23 * m35 * m44 * m52
                + m00 * m11 * m24 * m32 * m43 * m55
                - m00 * m11 * m24 * m32 * m45 * m53
                - m00 * m11 * m24 * m33 * m42 * m55
                + m00 * m11 * m24 * m33 * m45 * m52
                + m00 * m11 * m24 * m35 * m42 * m53
                - m00 * m11 * m24 * m35 * m43 * m52
                - m00 * m11 * m25 * m32 * m43 * m54
                + m00 * m11 * m25 * m32 * m44 * m53
                + m00 * m11 * m25 * m33 * m42 * m54
                - m00 * m11 * m25 * m33 * m44 * m52
                - m00 * m11 * m25 * m34 * m42 * m53
                + m00 * m11 * m25 * m34 * m43 * m52
                - m00 * m12 * m21 * m33 * m44 * m55
                + m00 * m12 * m21 * m33 * m45 * m54
                + m00 * m12 * m21 * m34 * m43 * m55
                - m00 * m12 * m21 * m34 * m45 * m53
                - m00 * m12 * m21 * m35 * m43 * m54
                + m00 * m12 * m21 * m35 * m44 * m53
                + m00 * m12 * m23 * m31 * m44 * m55
                - m00 * m12 * m23 * m31 * m45 * m54
                - m00 * m12 * m23 * m34 * m41 * m55
                + m00 * m12 * m23 * m34 * m45 * m51
                + m00 * m12 * m23 * m35 * m41 * m54
                - m00 * m12 * m23 * m35 * m44 * m51
                - m00 * m12 * m24 * m31 * m43 * m55
                + m00 * m12 * m24 * m31 * m45 * m53
                + m00 * m12 * m24 * m33 * m41 * m55
                - m00 * m12 * m24 * m33 * m45 * m51
                - m00 * m12 * m24 * m35 * m41 * m53
                + m00 * m12 * m24 * m35 * m43 * m51
                + m00 * m12 * m25 * m31 * m43 * m54
                - m00 * m12 * m25 * m31 * m44 * m53
                - m00 * m12 * m25 * m33 * m41 * m54
                + m00 * m12 * m25 * m33 * m44 * m51
                + m00 * m12 * m25 * m34 * m41 * m53
                - m00 * m12 * m25 * m34 * m43 * m51
                + m00 * m13 * m21 * m32 * m44 * m55
                - m00 * m13 * m21 * m32 * m45 * m54
                - m00 * m13 * m21 * m34 * m42 * m55
                + m00 * m13 * m21 * m34 * m45 * m52
                + m00 * m13 * m21 * m35 * m42 * m54
                - m00 * m13 * m21 * m35 * m44 * m52
                - m00 * m13 * m22 * m31 * m44 * m55
                + m00 * m13 * m22 * m31 * m45 * m54
                + m00 * m13 * m22 * m34 * m41 * m55
                - m00 * m13 * m22 * m34 * m45 * m51
                - m00 * m13 * m22 * m35 * m41 * m54
                + m00 * m13 * m22 * m35 * m44 * m51
                + m00 * m13 * m24 * m31 * m42 * m55
                - m00 * m13 * m24 * m31 * m45 * m52
                - m00 * m13 * m24 * m32 * m41 * m55
                + m00 * m13 * m24 * m32 * m45 * m51
                + m00 * m13 * m24 * m35 * m41 * m52
                - m00 * m13 * m24 * m35 * m42 * m51
                - m00 * m13 * m25 * m31 * m42 * m54
                + m00 * m13 * m25 * m31 * m44 * m52
                + m00 * m13 * m25 * m32 * m41 * m54
                - m00 * m13 * m25 * m32 * m44 * m51
                - m00 * m13 * m25 * m34 * m41 * m52
                + m00 * m13 * m25 * m34 * m42 * m51
                - m00 * m14 * m21 * m32 * m43 * m55
                + m00 * m14 * m21 * m32 * m45 * m53
                + m00 * m14 * m21 * m33 * m42 * m55
                - m00 * m14 * m21 * m33 * m45 * m52
                - m00 * m14 * m21 * m35 * m42 * m53
                + m00 * m14 * m21 * m35 * m43 * m52
                + m00 * m14 * m22 * m31 * m43 * m55
                - m00 * m14 * m22 * m31 * m45 * m53
                - m00 * m14 * m22 * m33 * m41 * m55
                + m00 * m14 * m22 * m33 * m45 * m51
                + m00 * m14 * m22 * m35 * m41 * m53
                - m00 * m14 * m22 * m35 * m43 * m51
                - m00 * m14 * m23 * m31 * m42 * m55
                + m00 * m14 * m23 * m31 * m45 * m52
                + m00 * m14 * m23 * m32 * m41 * m55
                - m00 * m14 * m23 * m32 * m45 * m51
                - m00 * m14 * m23 * m35 * m41 * m52
                + m00 * m14 * m23 * m35 * m42 * m51
                + m00 * m14 * m25 * m31 * m42 * m53
                - m00 * m14 * m25 * m31 * m43 * m52
                - m00 * m14 * m25 * m32 * m41 * m53
                + m00 * m14 * m25 * m32 * m43 * m51
                + m00 * m14 * m25 * m33 * m41 * m52
                - m00 * m14 * m25 * m33 * m42 * m51
                + m00 * m15 * m21 * m32 * m43 * m54
                - m00 * m15 * m21 * m32 * m44 * m53
                - m00 * m15 * m21 * m33 * m42 * m54
                + m00 * m15 * m21 * m33 * m44 * m52
                + m00 * m15 * m21 * m34 * m42 * m53
                - m00 * m15 * m21 * m34 * m43 * m52
                - m00 * m15 * m22 * m31 * m43 * m54
                + m00 * m15 * m22 * m31 * m44 * m53
                + m00 * m15 * m22 * m33 * m41 * m54
                - m00 * m15 * m22 * m33 * m44 * m51
                - m00 * m15 * m22 * m34 * m41 * m53
                + m00 * m15 * m22 * m34 * m43 * m51
                + m00 * m15 * m23 * m31 * m42 * m54
                - m00 * m15 * m23 * m31 * m44 * m52
                - m00 * m15 * m23 * m32 * m41 * m54
                + m00 * m15 * m23 * m32 * m44 * m51
                + m00 * m15 * m23 * m34 * m41 * m52
                - m00 * m15 * m23 * m34 * m42 * m51
                - m00 * m15 * m24 * m31 * m42 * m53
                + m00 * m15 * m24 * m31 * m43 * m52
                + m00 * m15 * m24 * m32 * m41 * m53
                - m00 * m15 * m24 * m32 * m43 * m51
                - m00 * m15 * m24 * m33 * m41 * m52
                + m00 * m15 * m24 * m33 * m42 * m51
                - m01 * m10 * m22 * m33 * m44 * m55
                + m01 * m10 * m22 * m33 * m45 * m54
                + m01 * m10 * m22 * m34 * m43 * m55
                - m01 * m10 * m22 * m34 * m45 * m53
                - m01 * m10 * m22 * m35 * m43 * m54
                + m01 * m10 * m22 * m35 * m44 * m53
                + m01 * m10 * m23 * m32 * m44 * m55
                - m01 * m10 * m23 * m32 * m45 * m54
                - m01 * m10 * m23 * m34 * m42 * m55
                + m01 * m10 * m23 * m34 * m45 * m52
                + m01 * m10 * m23 * m35 * m42 * m54
                - m01 * m10 * m23 * m35 * m44 * m52
                - m01 * m10 * m24 * m32 * m43 * m55
                + m01 * m10 * m24 * m32 * m45 * m53
                + m01 * m10 * m24 * m33 * m42 * m55
                - m01 * m10 * m24 * m33 * m45 * m52
                - m01 * m10 * m24 * m35 * m42 * m53
                + m01 * m10 * m24 * m35 * m43 * m52
                + m01 * m10 * m25 * m32 * m43 * m54
                - m01 * m10 * m25 * m32 * m44 * m53
                - m01 * m10 * m25 * m33 * m42 * m54
                + m01 * m10 * m25 * m33 * m44 * m52
                + m01 * m10 * m25 * m34 * m42 * m53
                - m01 * m10 * m25 * m34 * m43 * m52
                + m01 * m12 * m20 * m33 * m44 * m55
                - m01 * m12 * m20 * m33 * m45 * m54
                - m01 * m12 * m20 * m34 * m43 * m55
                + m01 * m12 * m20 * m34 * m45 * m53
                + m01 * m12 * m20 * m35 * m43 * m54
                - m01 * m12 * m20 * m35 * m44 * m53
                - m01 * m12 * m23 * m30 * m44 * m55
                + m01 * m12 * m23 * m30 * m45 * m54
                + m01 * m12 * m23 * m34 * m40 * m55
                - m01 * m12 * m23 * m34 * m45 * m50
                - m01 * m12 * m23 * m35 * m40 * m54
                + m01 * m12 * m23 * m35 * m44 * m50
                + m01 * m12 * m24 * m30 * m43 * m55
                - m01 * m12 * m24 * m30 * m45 * m53
                - m01 * m12 * m24 * m33 * m40 * m55
                + m01 * m12 * m24 * m33 * m45 * m50
                + m01 * m12 * m24 * m35 * m40 * m53
                - m01 * m12 * m24 * m35 * m43 * m50
                - m01 * m12 * m25 * m30 * m43 * m54
                + m01 * m12 * m25 * m30 * m44 * m53
                + m01 * m12 * m25 * m33 * m40 * m54
                - m01 * m12 * m25 * m33 * m44 * m50
                - m01 * m12 * m25 * m34 * m40 * m53
                + m01 * m12 * m25 * m34 * m43 * m50
                - m01 * m13 * m20 * m32 * m44 * m55
                + m01 * m13 * m20 * m32 * m45 * m54
                + m01 * m13 * m20 * m34 * m42 * m55
                - m01 * m13 * m20 * m34 * m45 * m52
                - m01 * m13 * m20 * m35 * m42 * m54
                + m01 * m13 * m20 * m35 * m44 * m52
                + m01 * m13 * m22 * m30 * m44 * m55
                - m01 * m13 * m22 * m30 * m45 * m54
                - m01 * m13 * m22 * m34 * m40 * m55
                + m01 * m13 * m22 * m34 * m45 * m50
                + m01 * m13 * m22 * m35 * m40 * m54
                - m01 * m13 * m22 * m35 * m44 * m50
                - m01 * m13 * m24 * m30 * m42 * m55
                + m01 * m13 * m24 * m30 * m45 * m52
                + m01 * m13 * m24 * m32 * m40 * m55
                - m01 * m13 * m24 * m32 * m45 * m50
                - m01 * m13 * m24 * m35 * m40 * m52
                + m01 * m13 * m24 * m35 * m42 * m50
                + m01 * m13 * m25 * m30 * m42 * m54
                - m01 * m13 * m25 * m30 * m44 * m52
                - m01 * m13 * m25 * m32 * m40 * m54
                + m01 * m13 * m25 * m32 * m44 * m50
                + m01 * m13 * m25 * m34 * m40 * m52
                - m01 * m13 * m25 * m34 * m42 * m50
                + m01 * m14 * m20 * m32 * m43 * m55
                - m01 * m14 * m20 * m32 * m45 * m53
                - m01 * m14 * m20 * m33 * m42 * m55
                + m01 * m14 * m20 * m33 * m45 * m52
                + m01 * m14 * m20 * m35 * m42 * m53
                - m01 * m14 * m20 * m35 * m43 * m52
                - m01 * m14 * m22 * m30 * m43 * m55
                + m01 * m14 * m22 * m30 * m45 * m53
                + m01 * m14 * m22 * m33 * m40 * m55
                - m01 * m14 * m22 * m33 * m45 * m50
                - m01 * m14 * m22 * m35 * m40 * m53
                + m01 * m14 * m22 * m35 * m43 * m50
                + m01 * m14 * m23 * m30 * m42 * m55
                - m01 * m14 * m23 * m30 * m45 * m52
                - m01 * m14 * m23 * m32 * m40 * m55
                + m01 * m14 * m23 * m32 * m45 * m50
                + m01 * m14 * m23 * m35 * m40 * m52
                - m01 * m14 * m23 * m35 * m42 * m50
                - m01 * m14 * m25 * m30 * m42 * m53
                + m01 * m14 * m25 * m30 * m43 * m52
                + m01 * m14 * m25 * m32 * m40 * m53
                - m01 * m14 * m25 * m32 * m43 * m50
                - m01 * m14 * m25 * m33 * m40 * m52
                + m01 * m14 * m25 * m33 * m42 * m50
                - m01 * m15 * m20 * m32 * m43 * m54
                + m01 * m15 * m20 * m32 * m44 * m53
                + m01 * m15 * m20 * m33 * m42 * m54
                - m01 * m15 * m20 * m33 * m44 * m52
                - m01 * m15 * m20 * m34 * m42 * m53
                + m01 * m15 * m20 * m34 * m43 * m52
                + m01 * m15 * m22 * m30 * m43 * m54
                - m01 * m15 * m22 * m30 * m44 * m53
                - m01 * m15 * m22 * m33 * m40 * m54
                + m01 * m15 * m22 * m33 * m44 * m50
                + m01 * m15 * m22 * m34 * m40 * m53
                - m01 * m15 * m22 * m34 * m43 * m50
                - m01 * m15 * m23 * m30 * m42 * m54
                + m01 * m15 * m23 * m30 * m44 * m52
                + m01 * m15 * m23 * m32 * m40 * m54
                - m01 * m15 * m23 * m32 * m44 * m50
                - m01 * m15 * m23 * m34 * m40 * m52
                + m01 * m15 * m23 * m34 * m42 * m50
                + m01 * m15 * m24 * m30 * m42 * m53
                - m01 * m15 * m24 * m30 * m43 * m52
                - m01 * m15 * m24 * m32 * m40 * m53
                + m01 * m15 * m24 * m32 * m43 * m50
                + m01 * m15 * m24 * m33 * m40 * m52
                - m01 * m15 * m24 * m33 * m42 * m50
                + m02 * m10 * m21 * m33 * m44 * m55
                - m02 * m10 * m21 * m33 * m45 * m54
                - m02 * m10 * m21 * m34 * m43 * m55
                + m02 * m10 * m21 * m34 * m45 * m53
                + m02 * m10 * m21 * m35 * m43 * m54
                - m02 * m10 * m21 * m35 * m44 * m53
                - m02 * m10 * m23 * m31 * m44 * m55
                + m02 * m10 * m23 * m31 * m45 * m54
                + m02 * m10 * m23 * m34 * m41 * m55
                - m02 * m10 * m23 * m34 * m45 * m51
                - m02 * m10 * m23 * m35 * m41 * m54
                + m02 * m10 * m23 * m35 * m44 * m51
                + m02 * m10 * m24 * m31 * m43 * m55
                - m02 * m10 * m24 * m31 * m45 * m53
                - m02 * m10 * m24 * m33 * m41 * m55
                + m02 * m10 * m24 * m33 * m45 * m51
                + m02 * m10 * m24 * m35 * m41 * m53
                - m02 * m10 * m24 * m35 * m43 * m51
                - m02 * m10 * m25 * m31 * m43 * m54
                + m02 * m10 * m25 * m31 * m44 * m53
                + m02 * m10 * m25 * m33 * m41 * m54
                - m02 * m10 * m25 * m33 * m44 * m51
                - m02 * m10 * m25 * m34 * m41 * m53
                + m02 * m10 * m25 * m34 * m43 * m51
                - m02 * m11 * m20 * m33 * m44 * m55
                + m02 * m11 * m20 * m33 * m45 * m54
                + m02 * m11 * m20 * m34 * m43 * m55
                - m02 * m11 * m20 * m34 * m45 * m53
                - m02 * m11 * m20 * m35 * m43 * m54
                + m02 * m11 * m20 * m35 * m44 * m53
                + m02 * m11 * m23 * m30 * m44 * m55
                - m02 * m11 * m23 * m30 * m45 * m54
                - m02 * m11 * m23 * m34 * m40 * m55
                + m02 * m11 * m23 * m34 * m45 * m50
                + m02 * m11 * m23 * m35 * m40 * m54
                - m02 * m11 * m23 * m35 * m44 * m50
                - m02 * m11 * m24 * m30 * m43 * m55
                + m02 * m11 * m24 * m30 * m45 * m53
                + m02 * m11 * m24 * m33 * m40 * m55
                - m02 * m11 * m24 * m33 * m45 * m50
                - m02 * m11 * m24 * m35 * m40 * m53
                + m02 * m11 * m24 * m35 * m43 * m50
                + m02 * m11 * m25 * m30 * m43 * m54
                - m02 * m11 * m25 * m30 * m44 * m53
                - m02 * m11 * m25 * m33 * m40 * m54
                + m02 * m11 * m25 * m33 * m44 * m50
                + m02 * m11 * m25 * m34 * m40 * m53
                - m02 * m11 * m25 * m34 * m43 * m50
                + m02 * m13 * m20 * m31 * m44 * m55
                - m02 * m13 * m20 * m31 * m45 * m54
                - m02 * m13 * m20 * m34 * m41 * m55
                + m02 * m13 * m20 * m34 * m45 * m51
                + m02 * m13 * m20 * m35 * m41 * m54
                - m02 * m13 * m20 * m35 * m44 * m51
                - m02 * m13 * m21 * m30 * m44 * m55
                + m02 * m13 * m21 * m30 * m45 * m54
                + m02 * m13 * m21 * m34 * m40 * m55
                - m02 * m13 * m21 * m34 * m45 * m50
                - m02 * m13 * m21 * m35 * m40 * m54
                + m02 * m13 * m21 * m35 * m44 * m50
                + m02 * m13 * m24 * m30 * m41 * m55
                - m02 * m13 * m24 * m30 * m45 * m51
                - m02 * m13 * m24 * m31 * m40 * m55
                + m02 * m13 * m24 * m31 * m45 * m50
                + m02 * m13 * m24 * m35 * m40 * m51
                - m02 * m13 * m24 * m35 * m41 * m50
                - m02 * m13 * m25 * m30 * m41 * m54
                + m02 * m13 * m25 * m30 * m44 * m51
                + m02 * m13 * m25 * m31 * m40 * m54
                - m02 * m13 * m25 * m31 * m44 * m50
                - m02 * m13 * m25 * m34 * m40 * m51
                + m02 * m13 * m25 * m34 * m41 * m50
                - m02 * m14 * m20 * m31 * m43 * m55
                + m02 * m14 * m20 * m31 * m45 * m53
                + m02 * m14 * m20 * m33 * m41 * m55
                - m02 * m14 * m20 * m33 * m45 * m51
                - m02 * m14 * m20 * m35 * m41 * m53
                + m02 * m14 * m20 * m35 * m43 * m51
                + m02 * m14 * m21 * m30 * m43 * m55
                - m02 * m14 * m21 * m30 * m45 * m53
                - m02 * m14 * m21 * m33 * m40 * m55
                + m02 * m14 * m21 * m33 * m45 * m50
                + m02 * m14 * m21 * m35 * m40 * m53
                - m02 * m14 * m21 * m35 * m43 * m50
                - m02 * m14 * m23 * m30 * m41 * m55
                + m02 * m14 * m23 * m30 * m45 * m51
                + m02 * m14 * m23 * m31 * m40 * m55
                - m02 * m14 * m23 * m31 * m45 * m50
                - m02 * m14 * m23 * m35 * m40 * m51
                + m02 * m14 * m23 * m35 * m41 * m50
                + m02 * m14 * m25 * m30 * m41 * m53
                - m02 * m14 * m25 * m30 * m43 * m51
                - m02 * m14 * m25 * m31 * m40 * m53
                + m02 * m14 * m25 * m31 * m43 * m50
                + m02 * m14 * m25 * m33 * m40 * m51
                - m02 * m14 * m25 * m33 * m41 * m50
                + m02 * m15 * m20 * m31 * m43 * m54
                - m02 * m15 * m20 * m31 * m44 * m53
                - m02 * m15 * m20 * m33 * m41 * m54
                + m02 * m15 * m20 * m33 * m44 * m51
                + m02 * m15 * m20 * m34 * m41 * m53
                - m02 * m15 * m20 * m34 * m43 * m51
                - m02 * m15 * m21 * m30 * m43 * m54
                + m02 * m15 * m21 * m30 * m44 * m53
                + m02 * m15 * m21 * m33 * m40 * m54
                - m02 * m15 * m21 * m33 * m44 * m50
                - m02 * m15 * m21 * m34 * m40 * m53
                + m02 * m15 * m21 * m34 * m43 * m50
                + m02 * m15 * m23 * m30 * m41 * m54
                - m02 * m15 * m23 * m30 * m44 * m51
                - m02 * m15 * m23 * m31 * m40 * m54
                + m02 * m15 * m23 * m31 * m44 * m50
                + m02 * m15 * m23 * m34 * m40 * m51
                - m02 * m15 * m23 * m34 * m41 * m50
                - m02 * m15 * m24 * m30 * m41 * m53
                + m02 * m15 * m24 * m30 * m43 * m51
                + m02 * m15 * m24 * m31 * m40 * m53
                - m02 * m15 * m24 * m31 * m43 * m50
                - m02 * m15 * m24 * m33 * m40 * m51
                + m02 * m15 * m24 * m33 * m41 * m50
                - m03 * m10 * m21 * m32 * m44 * m55
                + m03 * m10 * m21 * m32 * m45 * m54
                + m03 * m10 * m21 * m34 * m42 * m55
                - m03 * m10 * m21 * m34 * m45 * m52
                - m03 * m10 * m21 * m35 * m42 * m54
                + m03 * m10 * m21 * m35 * m44 * m52
                + m03 * m10 * m22 * m31 * m44 * m55
                - m03 * m10 * m22 * m31 * m45 * m54
                - m03 * m10 * m22 * m34 * m41 * m55
                + m03 * m10 * m22 * m34 * m45 * m51
                + m03 * m10 * m22 * m35 * m41 * m54
                - m03 * m10 * m22 * m35 * m44 * m51
                - m03 * m10 * m24 * m31 * m42 * m55
                + m03 * m10 * m24 * m31 * m45 * m52
                + m03 * m10 * m24 * m32 * m41 * m55
                - m03 * m10 * m24 * m32 * m45 * m51
                - m03 * m10 * m24 * m35 * m41 * m52
                + m03 * m10 * m24 * m35 * m42 * m51
                + m03 * m10 * m25 * m31 * m42 * m54
                - m03 * m10 * m25 * m31 * m44 * m52
                - m03 * m10 * m25 * m32 * m41 * m54
                + m03 * m10 * m25 * m32 * m44 * m51
                + m03 * m10 * m25 * m34 * m41 * m52
                - m03 * m10 * m25 * m34 * m42 * m51
                + m03 * m11 * m20 * m32 * m44 * m55
                - m03 * m11 * m20 * m32 * m45 * m54
                - m03 * m11 * m20 * m34 * m42 * m55
                + m03 * m11 * m20 * m34 * m45 * m52
                + m03 * m11 * m20 * m35 * m42 * m54
                - m03 * m11 * m20 * m35 * m44 * m52
                - m03 * m11 * m22 * m30 * m44 * m55
                + m03 * m11 * m22 * m30 * m45 * m54
                + m03 * m11 * m22 * m34 * m40 * m55
                - m03 * m11 * m22 * m34 * m45 * m50
                - m03 * m11 * m22 * m35 * m40 * m54
                + m03 * m11 * m22 * m35 * m44 * m50
                + m03 * m11 * m24 * m30 * m42 * m55
                - m03 * m11 * m24 * m30 * m45 * m52
                - m03 * m11 * m24 * m32 * m40 * m55
                + m03 * m11 * m24 * m32 * m45 * m50
                + m03 * m11 * m24 * m35 * m40 * m52
                - m03 * m11 * m24 * m35 * m42 * m50
                - m03 * m11 * m25 * m30 * m42 * m54
                + m03 * m11 * m25 * m30 * m44 * m52
                + m03 * m11 * m25 * m32 * m40 * m54
                - m03 * m11 * m25 * m32 * m44 * m50
                - m03 * m11 * m25 * m34 * m40 * m52
                + m03 * m11 * m25 * m34 * m42 * m50
                - m03 * m12 * m20 * m31 * m44 * m55
                + m03 * m12 * m20 * m31 * m45 * m54
                + m03 * m12 * m20 * m34 * m41 * m55
                - m03 * m12 * m20 * m34 * m45 * m51
                - m03 * m12 * m20 * m35 * m41 * m54
                + m03 * m12 * m20 * m35 * m44 * m51
                + m03 * m12 * m21 * m30 * m44 * m55
                - m03 * m12 * m21 * m30 * m45 * m54
                - m03 * m12 * m21 * m34 * m40 * m55
                + m03 * m12 * m21 * m34 * m45 * m50
                + m03 * m12 * m21 * m35 * m40 * m54
                - m03 * m12 * m21 * m35 * m44 * m50
                - m03 * m12 * m24 * m30 * m41 * m55
                + m03 * m12 * m24 * m30 * m45 * m51
                + m03 * m12 * m24 * m31 * m40 * m55
                - m03 * m12 * m24 * m31 * m45 * m50
                - m03 * m12 * m24 * m35 * m40 * m51
                + m03 * m12 * m24 * m35 * m41 * m50
                + m03 * m12 * m25 * m30 * m41 * m54
                - m03 * m12 * m25 * m30 * m44 * m51
                - m03 * m12 * m25 * m31 * m40 * m54
                + m03 * m12 * m25 * m31 * m44 * m50
                + m03 * m12 * m25 * m34 * m40 * m51
                - m03 * m12 * m25 * m34 * m41 * m50
                + m03 * m14 * m20 * m31 * m42 * m55
                - m03 * m14 * m20 * m31 * m45 * m52
                - m03 * m14 * m20 * m32 * m41 * m55
                + m03 * m14 * m20 * m32 * m45 * m51
                + m03 * m14 * m20 * m35 * m41 * m52
                - m03 * m14 * m20 * m35 * m42 * m51
                - m03 * m14 * m21 * m30 * m42 * m55
                + m03 * m14 * m21 * m30 * m45 * m52
                + m03 * m14 * m21 * m32 * m40 * m55
                - m03 * m14 * m21 * m32 * m45 * m50
                - m03 * m14 * m21 * m35 * m40 * m52
                + m03 * m14 * m21 * m35 * m42 * m50
                + m03 * m14 * m22 * m30 * m41 * m55
                - m03 * m14 * m22 * m30 * m45 * m51
                - m03 * m14 * m22 * m31 * m40 * m55
                + m03 * m14 * m22 * m31 * m45 * m50
                + m03 * m14 * m22 * m35 * m40 * m51
                - m03 * m14 * m22 * m35 * m41 * m50
                - m03 * m14 * m25 * m30 * m41 * m52
                + m03 * m14 * m25 * m30 * m42 * m51
                + m03 * m14 * m25 * m31 * m40 * m52
                - m03 * m14 * m25 * m31 * m42 * m50
                - m03 * m14 * m25 * m32 * m40 * m51
                + m03 * m14 * m25 * m32 * m41 * m50
                - m03 * m15 * m20 * m31 * m42 * m54
                + m03 * m15 * m20 * m31 * m44 * m52
                + m03 * m15 * m20 * m32 * m41 * m54
                - m03 * m15 * m20 * m32 * m44 * m51
                - m03 * m15 * m20 * m34 * m41 * m52
                + m03 * m15 * m20 * m34 * m42 * m51
                + m03 * m15 * m21 * m30 * m42 * m54
                - m03 * m15 * m21 * m30 * m44 * m52
                - m03 * m15 * m21 * m32 * m40 * m54
                + m03 * m15 * m21 * m32 * m44 * m50
                + m03 * m15 * m21 * m34 * m40 * m52
                - m03 * m15 * m21 * m34 * m42 * m50
                - m03 * m15 * m22 * m30 * m41 * m54
                + m03 * m15 * m22 * m30 * m44 * m51
                + m03 * m15 * m22 * m31 * m40 * m54
                - m03 * m15 * m22 * m31 * m44 * m50
                - m03 * m15 * m22 * m34 * m40 * m51
                + m03 * m15 * m22 * m34 * m41 * m50
                + m03 * m15 * m24 * m30 * m41 * m52
                - m03 * m15 * m24 * m30 * m42 * m51
                - m03 * m15 * m24 * m31 * m40 * m52
                + m03 * m15 * m24 * m31 * m42 * m50
                + m03 * m15 * m24 * m32 * m40 * m51
                - m03 * m15 * m24 * m32 * m41 * m50
                + m04 * m10 * m21 * m32 * m43 * m55
                - m04 * m10 * m21 * m32 * m45 * m53
                - m04 * m10 * m21 * m33 * m42 * m55
                + m04 * m10 * m21 * m33 * m45 * m52
                + m04 * m10 * m21 * m35 * m42 * m53
                - m04 * m10 * m21 * m35 * m43 * m52
                - m04 * m10 * m22 * m31 * m43 * m55
                + m04 * m10 * m22 * m31 * m45 * m53
                + m04 * m10 * m22 * m33 * m41 * m55
                - m04 * m10 * m22 * m33 * m45 * m51
                - m04 * m10 * m22 * m35 * m41 * m53
                + m04 * m10 * m22 * m35 * m43 * m51
                + m04 * m10 * m23 * m31 * m42 * m55
                - m04 * m10 * m23 * m31 * m45 * m52
                - m04 * m10 * m23 * m32 * m41 * m55
                + m04 * m10 * m23 * m32 * m45 * m51
                + m04 * m10 * m23 * m35 * m41 * m52
                - m04 * m10 * m23 * m35 * m42 * m51
                - m04 * m10 * m25 * m31 * m42 * m53
                + m04 * m10 * m25 * m31 * m43 * m52
                + m04 * m10 * m25 * m32 * m41 * m53
                - m04 * m10 * m25 * m32 * m43 * m51
                - m04 * m10 * m25 * m33 * m41 * m52
                + m04 * m10 * m25 * m33 * m42 * m51
                - m04 * m11 * m20 * m32 * m43 * m55
                + m04 * m11 * m20 * m32 * m45 * m53
                + m04 * m11 * m20 * m33 * m42 * m55
                - m04 * m11 * m20 * m33 * m45 * m52
                - m04 * m11 * m20 * m35 * m42 * m53
                + m04 * m11 * m20 * m35 * m43 * m52
                + m04 * m11 * m22 * m30 * m43 * m55
                - m04 * m11 * m22 * m30 * m45 * m53
                - m04 * m11 * m22 * m33 * m40 * m55
                + m04 * m11 * m22 * m33 * m45 * m50
                + m04 * m11 * m22 * m35 * m40 * m53
                - m04 * m11 * m22 * m35 * m43 * m50
                - m04 * m11 * m23 * m30 * m42 * m55
                + m04 * m11 * m23 * m30 * m45 * m52
                + m04 * m11 * m23 * m32 * m40 * m55
                - m04 * m11 * m23 * m32 * m45 * m50
                - m04 * m11 * m23 * m35 * m40 * m52
                + m04 * m11 * m23 * m35 * m42 * m50
                + m04 * m11 * m25 * m30 * m42 * m53
                - m04 * m11 * m25 * m30 * m43 * m52
                - m04 * m11 * m25 * m32 * m40 * m53
                + m04 * m11 * m25 * m32 * m43 * m50
                + m04 * m11 * m25 * m33 * m40 * m52
                - m04 * m11 * m25 * m33 * m42 * m50
                + m04 * m12 * m20 * m31 * m43 * m55
                - m04 * m12 * m20 * m31 * m45 * m53
                - m04 * m12 * m20 * m33 * m41 * m55
                + m04 * m12 * m20 * m33 * m45 * m51
                + m04 * m12 * m20 * m35 * m41 * m53
                - m04 * m12 * m20 * m35 * m43 * m51
                - m04 * m12 * m21 * m30 * m43 * m55
                + m04 * m12 * m21 * m30 * m45 * m53
                + m04 * m12 * m21 * m33 * m40 * m55
                - m04 * m12 * m21 * m33 * m45 * m50
                - m04 * m12 * m21 * m35 * m40 * m53
                + m04 * m12 * m21 * m35 * m43 * m50
                + m04 * m12 * m23 * m30 * m41 * m55
                - m04 * m12 * m23 * m30 * m45 * m51
                - m04 * m12 * m23 * m31 * m40 * m55
                + m04 * m12 * m23 * m31 * m45 * m50
                + m04 * m12 * m23 * m35 * m40 * m51
                - m04 * m12 * m23 * m35 * m41 * m50
                - m04 * m12 * m25 * m30 * m41 * m53
                + m04 * m12 * m25 * m30 * m43 * m51
                + m04 * m12 * m25 * m31 * m40 * m53
                - m04 * m12 * m25 * m31 * m43 * m50
                - m04 * m12 * m25 * m33 * m40 * m51
                + m04 * m12 * m25 * m33 * m41 * m50
                - m04 * m13 * m20 * m31 * m42 * m55
                + m04 * m13 * m20 * m31 * m45 * m52
                + m04 * m13 * m20 * m32 * m41 * m55
                - m04 * m13 * m20 * m32 * m45 * m51
                - m04 * m13 * m20 * m35 * m41 * m52
                + m04 * m13 * m20 * m35 * m42 * m51
                + m04 * m13 * m21 * m30 * m42 * m55
                - m04 * m13 * m21 * m30 * m45 * m52
                - m04 * m13 * m21 * m32 * m40 * m55
                + m04 * m13 * m21 * m32 * m45 * m50
                + m04 * m13 * m21 * m35 * m40 * m52
                - m04 * m13 * m21 * m35 * m42 * m50
                - m04 * m13 * m22 * m30 * m41 * m55
                + m04 * m13 * m22 * m30 * m45 * m51
                + m04 * m13 * m22 * m31 * m40 * m55
                - m04 * m13 * m22 * m31 * m45 * m50
                - m04 * m13 * m22 * m35 * m40 * m51
                + m04 * m13 * m22 * m35 * m41 * m50
                + m04 * m13 * m25 * m30 * m41 * m52
                - m04 * m13 * m25 * m30 * m42 * m51
                - m04 * m13 * m25 * m31 * m40 * m52
                + m04 * m13 * m25 * m31 * m42 * m50
                + m04 * m13 * m25 * m32 * m40 * m51
                - m04 * m13 * m25 * m32 * m41 * m50
                + m04 * m15 * m20 * m31 * m42 * m53
                - m04 * m15 * m20 * m31 * m43 * m52
                - m04 * m15 * m20 * m32 * m41 * m53
                + m04 * m15 * m20 * m32 * m43 * m51
                + m04 * m15 * m20 * m33 * m41 * m52
                - m04 * m15 * m20 * m33 * m42 * m51
                - m04 * m15 * m21 * m30 * m42 * m53
                + m04 * m15 * m21 * m30 * m43 * m52
                + m04 * m15 * m21 * m32 * m40 * m53
                - m04 * m15 * m21 * m32 * m43 * m50
                - m04 * m15 * m21 * m33 * m40 * m52
                + m04 * m15 * m21 * m33 * m42 * m50
                + m04 * m15 * m22 * m30 * m41 * m53
                - m04 * m15 * m22 * m30 * m43 * m51
                - m04 * m15 * m22 * m31 * m40 * m53
                + m04 * m15 * m22 * m31 * m43 * m50
                + m04 * m15 * m22 * m33 * m40 * m51
                - m04 * m15 * m22 * m33 * m41 * m50
                - m04 * m15 * m23 * m30 * m41 * m52
                + m04 * m15 * m23 * m30 * m42 * m51
                + m04 * m15 * m23 * m31 * m40 * m52
                - m04 * m15 * m23 * m31 * m42 * m50
                - m04 * m15 * m23 * m32 * m40 * m51
                + m04 * m15 * m23 * m32 * m41 * m50
                - m05 * m10 * m21 * m32 * m43 * m54
                + m05 * m10 * m21 * m32 * m44 * m53
                + m05 * m10 * m21 * m33 * m42 * m54
                - m05 * m10 * m21 * m33 * m44 * m52
                - m05 * m10 * m21 * m34 * m42 * m53
                + m05 * m10 * m21 * m34 * m43 * m52
                + m05 * m10 * m22 * m31 * m43 * m54
                - m05 * m10 * m22 * m31 * m44 * m53
                - m05 * m10 * m22 * m33 * m41 * m54
                + m05 * m10 * m22 * m33 * m44 * m51
                + m05 * m10 * m22 * m34 * m41 * m53
                - m05 * m10 * m22 * m34 * m43 * m51
                - m05 * m10 * m23 * m31 * m42 * m54
                + m05 * m10 * m23 * m31 * m44 * m52
                + m05 * m10 * m23 * m32 * m41 * m54
                - m05 * m10 * m23 * m32 * m44 * m51
                - m05 * m10 * m23 * m34 * m41 * m52
                + m05 * m10 * m23 * m34 * m42 * m51
                + m05 * m10 * m24 * m31 * m42 * m53
                - m05 * m10 * m24 * m31 * m43 * m52
                - m05 * m10 * m24 * m32 * m41 * m53
                + m05 * m10 * m24 * m32 * m43 * m51
                + m05 * m10 * m24 * m33 * m41 * m52
                - m05 * m10 * m24 * m33 * m42 * m51
                + m05 * m11 * m20 * m32 * m43 * m54
                - m05 * m11 * m20 * m32 * m44 * m53
                - m05 * m11 * m20 * m33 * m42 * m54
                + m05 * m11 * m20 * m33 * m44 * m52
                + m05 * m11 * m20 * m34 * m42 * m53
                - m05 * m11 * m20 * m34 * m43 * m52
                - m05 * m11 * m22 * m30 * m43 * m54
                + m05 * m11 * m22 * m30 * m44 * m53
                + m05 * m11 * m22 * m33 * m40 * m54
                - m05 * m11 * m22 * m33 * m44 * m50
                - m05 * m11 * m22 * m34 * m40 * m53
                + m05 * m11 * m22 * m34 * m43 * m50
                + m05 * m11 * m23 * m30 * m42 * m54
                - m05 * m11 * m23 * m30 * m44 * m52
                - m05 * m11 * m23 * m32 * m40 * m54
                + m05 * m11 * m23 * m32 * m44 * m50
                + m05 * m11 * m23 * m34 * m40 * m52
                - m05 * m11 * m23 * m34 * m42 * m50
                - m05 * m11 * m24 * m30 * m42 * m53
                + m05 * m11 * m24 * m30 * m43 * m52
                + m05 * m11 * m24 * m32 * m40 * m53
                - m05 * m11 * m24 * m32 * m43 * m50
                - m05 * m11 * m24 * m33 * m40 * m52
                + m05 * m11 * m24 * m33 * m42 * m50
                - m05 * m12 * m20 * m31 * m43 * m54
                + m05 * m12 * m20 * m31 * m44 * m53
                + m05 * m12 * m20 * m33 * m41 * m54
                - m05 * m12 * m20 * m33 * m44 * m51
                - m05 * m12 * m20 * m34 * m41 * m53
                + m05 * m12 * m20 * m34 * m43 * m51
                + m05 * m12 * m21 * m30 * m43 * m54
                - m05 * m12 * m21 * m30 * m44 * m53
                - m05 * m12 * m21 * m33 * m40 * m54
                + m05 * m12 * m21 * m33 * m44 * m50
                + m05 * m12 * m21 * m34 * m40 * m53
                - m05 * m12 * m21 * m34 * m43 * m50
                - m05 * m12 * m23 * m30 * m41 * m54
                + m05 * m12 * m23 * m30 * m44 * m51
                + m05 * m12 * m23 * m31 * m40 * m54
                - m05 * m12 * m23 * m31 * m44 * m50
                - m05 * m12 * m23 * m34 * m40 * m51
                + m05 * m12 * m23 * m34 * m41 * m50
                + m05 * m12 * m24 * m30 * m41 * m53
                - m05 * m12 * m24 * m30 * m43 * m51
                - m05 * m12 * m24 * m31 * m40 * m53
                + m05 * m12 * m24 * m31 * m43 * m50
                + m05 * m12 * m24 * m33 * m40 * m51
                - m05 * m12 * m24 * m33 * m41 * m50
                + m05 * m13 * m20 * m31 * m42 * m54
                - m05 * m13 * m20 * m31 * m44 * m52
                - m05 * m13 * m20 * m32 * m41 * m54
                + m05 * m13 * m20 * m32 * m44 * m51
                + m05 * m13 * m20 * m34 * m41 * m52
                - m05 * m13 * m20 * m34 * m42 * m51
                - m05 * m13 * m21 * m30 * m42 * m54
                + m05 * m13 * m21 * m30 * m44 * m52
                + m05 * m13 * m21 * m32 * m40 * m54
                - m05 * m13 * m21 * m32 * m44 * m50
                - m05 * m13 * m21 * m34 * m40 * m52
                + m05 * m13 * m21 * m34 * m42 * m50
                + m05 * m13 * m22 * m30 * m41 * m54
                - m05 * m13 * m22 * m30 * m44 * m51
                - m05 * m13 * m22 * m31 * m40 * m54
                + m05 * m13 * m22 * m31 * m44 * m50
                + m05 * m13 * m22 * m34 * m40 * m51
                - m05 * m13 * m22 * m34 * m41 * m50
                - m05 * m13 * m24 * m30 * m41 * m52
                + m05 * m13 * m24 * m30 * m42 * m51
                + m05 * m13 * m24 * m31 * m40 * m52
                - m05 * m13 * m24 * m31 * m42 * m50
                - m05 * m13 * m24 * m32 * m40 * m51
                + m05 * m13 * m24 * m32 * m41 * m50
                - m05 * m14 * m20 * m31 * m42 * m53
                + m05 * m14 * m20 * m31 * m43 * m52
                + m05 * m14 * m20 * m32 * m41 * m53
                - m05 * m14 * m20 * m32 * m43 * m51
                - m05 * m14 * m20 * m33 * m41 * m52
                + m05 * m14 * m20 * m33 * m42 * m51
                + m05 * m14 * m21 * m30 * m42 * m53
                - m05 * m14 * m21 * m30 * m43 * m52
                - m05 * m14 * m21 * m32 * m40 * m53
                + m05 * m14 * m21 * m32 * m43 * m50
                + m05 * m14 * m21 * m33 * m40 * m52
                - m05 * m14 * m21 * m33 * m42 * m50
                - m05 * m14 * m22 * m30 * m41 * m53
                + m05 * m14 * m22 * m30 * m43 * m51
                + m05 * m14 * m22 * m31 * m40 * m53
                - m05 * m14 * m22 * m31 * m43 * m50
                - m05 * m14 * m22 * m33 * m40 * m51
                + m05 * m14 * m22 * m33 * m41 * m50
                + m05 * m14 * m23 * m30 * m41 * m52
                - m05 * m14 * m23 * m30 * m42 * m51
                - m05 * m14 * m23 * m31 * m40 * m52
                + m05 * m14 * m23 * m31 * m42 * m50
                + m05 * m14 * m23 * m32 * m40 * m51
                - m05 * m14 * m23 * m32 * m41 * m50
        }
        _ =>
        // Now do it the traditional way
        {
            // Reuseable result to reduce memory allocations
            let mut res = Array2::<T>::zeros((l - 1, l - 1));
            (0..l)
                .map(|i| {
                    for r in 0..l - 1 {
                        for c in 0..l - 1 {
                            res[(r, c)] = 
                                if c < i {
                                    m[(r + 1, c)]
                                } else {
                                    m[(r + 1, c + 1)]
                                };
                        }
                    }
                    let v = m[(0, i)] * determinant(&res);
                    if (i % 2) == 0 {
                        v
                    } else {
                        -v
                    }
                })
                .sum::<T>()
        }
    }
}

#[cfg(test)]
mod test_inverse {
    use ndarray::prelude::*;
    use crate::Inverse;
    use num_traits::Float;
    use std::fmt::Debug;

    const EPS: f64 = 1e-8;

    fn to_vec<T: Float>(m: &Array2<T>) -> Vec<T> {
        m.indexed_iter().map(|(_, &e)| e).collect()
    }

    /// utility function to compare vectors of Floats
    fn compare_vecs<T: Float + Debug>(v1: &Array2<T>, v2: &Array2<T>, epsilon: T) -> bool {

        to_vec(v1)
            .into_iter()
            .zip(to_vec(v2))
            .map(|(a, b)| Float::abs(a.abs() - b.abs()) < epsilon)
            .all(|b| b)
    }

    #[test]
    fn test_1x1() {
        let a: Array2<f64> = array![[2.0]];
        let inv = a.inv().expect("Sods Law");
        let back = inv.inv();
        assert!(compare_vecs(&a, &back.unwrap(), EPS));
        let expected = array![[0.5]];
        assert!(compare_vecs(&inv, &expected, EPS));
    }

    #[test]
    fn test_2x2() {
        let a: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
        let inv = a.inv().expect("Sods Law");
        let back = inv.inv();
        assert!(compare_vecs(&a, &back.unwrap(), EPS));
        let expected = array![[-2.0, 1.0], [1.5, -0.5]];
        assert!(compare_vecs(&inv, &expected, EPS));
    }

    #[test]
    fn test_3x3() {
        let a: Array2<f64> = array![[1.0, 0.0, 3.0], [2.0, 1.0, 6.0], [1.0, 0.0, 9.0]];
        let inv = a.inv().expect("Sods Law");
        let back = inv.inv();
        assert!(compare_vecs(&a, &back.unwrap(), EPS));
        let expected = array![[1.5, 0.0, -0.5], [-2.0, 1.0, -0.0], [-0.16666667, 0.0, 0.16666667]];
        assert!(compare_vecs(&inv, &expected, EPS));
    }

    #[test]
    fn test_4x4() {
        let a: Array2<f64> = array![
            [-68.0, 68.0, -16.0, 4.0],
            [-36.0, 35.0, -9.0, 3.0],
            [48.0, -47.0, 11.0, -3.0],
            [64.0, -64.0, 16.0, -4.0]
        ];
        let inv = a.inv().expect("Sods Law");
        let back = inv.inv();
        assert!(compare_vecs(&a, &back.unwrap(), EPS));
        let expected = array![
            [1.25, 0.5, 1.5, 0.5],
            [1.5, 0.5, 1.5, 0.75],
            [1.5, 0.5, 0.5, 1.5],
            [2.0, 2.0, 2.0, 1.75]
        ];
        assert!(compare_vecs(&inv, &expected, EPS));
    }

    #[test]
    fn test_5x5() {
        let a: Array2<f64> = array![
            [10.0, 1.0, 7.0, 1.0, 5.0],
            [2.0, 4.0, 8.0, 3.0, 2.0],
            [5.0, 1.0, 2.0, 9.0, 10.0],
            [6.0, 9.0, 9.0, 7.0, 3.0],
            [1.0, 8.0, 8.0, 10.0, 5.0],
        ];
        let inv = a.inv().expect("Sods Law");
        let back = inv.inv().unwrap();
        // some concern here, precision not ideal
        eprintln!("{:?}", to_vec(&a));
        eprintln!("{:?}", to_vec(&back));
        assert!(compare_vecs(&a, &back, 1e-5));
        let expected: Array2<f64> = array![
            [-11.98,  15.64,   9.32,  10.34, -19.12],
            [ 33.62, -44.16, -26.08, -28.46,  53.28],
            [ -9.36,  12.48,   7.24,   7.88, -14.84],
            [-37.2 ,  48.6 ,  28.8 ,  31.6 , -58.8 ],
            [ 37.98, -49.64, -29.32, -32.34,  60.12]
        ];
        assert!(compare_vecs(&inv, &expected, EPS));
    }

    #[test]
    fn test_6x6() {
        let a: Array2<f64> = array![
            [1.0, 1.0, 3.0, 4.0, 9.0, 3.0],
            [10.0, 10.0, 1.0, 2.0, 2.0, 5.0],
            [2.0, 9.0, 6.0, 10.0, 10.0, 9.0],
            [10.0, 9.0, 9.0, 7.0, 3.0, 6.0],
            [7.0, 6.0, 6.0, 2.0, 9.0, 5.0],
            [3.0, 8.0, 1.0, 4.0, 1.0, 5.0]
        ];
        let inv = a.inv().expect("Sods Law");
        let back = inv.inv();
        assert!(compare_vecs(&a, &back.unwrap(), EPS));
        let expected = array![
            [-0.53881418,  0.57793399,  0.6653423 , -0.08374083, -0.16962103,
             -1.18215159],
            [ 2.16075795, -1.52750611, -2.44070905,  0.44132029,  0.32457213,
              3.77017115],
            [ 0.21454768, -0.41503667, -0.39425428,  0.14792176,  0.19743276,
              0.62102689],
            [ 0.70018337, -0.24052567, -0.525978  ,  0.20354523, -0.21179707,
              0.73471883],
            [ 0.85055012, -0.47157702, -0.82793399,  0.1106357 ,  0.1146088 ,
              1.20415648],
            [-3.90709046,  2.46699267,  4.17114914, -0.87041565, -0.31051345,
             -6.07579462]
        ];
        assert!(compare_vecs(&inv, &expected, EPS));
    }

    #[cfg(not(debug_assertions))]
    use std::time::{Instant, Duration};

    #[test]
    #[cfg(not(debug_assertions))]
    // This is very fast when compiled with full optimisations
    // About 50x slower in debug mode! Thanks Optimizer.
    fn bench_6x6() {
        let a: Array2<f64> = array![
            [1.0, 1.0, 3.0, 4.0, 9.0, 3.0],
            [10.0, 10.0, 1.0, 2.0, 2.0, 5.0],
            [2.0, 9.0, 6.0, 10.0, 10.0, 9.0],
            [10.0, 9.0, 9.0, 7.0, 3.0, 6.0],
            [7.0, 6.0, 6.0, 2.0, 9.0, 5.0],
            [3.0, 8.0, 1.0, 4.0, 1.0, 5.0]
        ];

        let start = Instant::now();
        let mut _q = a.inv();
        for _ in 0..100000 {
            _q = a.inv();
        }
        // Takes about a second on average modern desktop / laptop
        assert!(start.elapsed() < Duration::new(2, 0));
    }
    /*
    */
}
