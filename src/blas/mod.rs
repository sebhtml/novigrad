mod cblas;
pub use cblas::*;

pub enum Layout {
    RowMajor,
    ColumnMajor,
}

pub enum Transpose {
    None,
    Ordinary,
    Conjugate,
}

pub trait BlasMethods {
    /// SGEMM  performs one of the matrix-matrix operations
    fn sgemm(
        layout: Layout,
        transa: Transpose,
        transb: Transpose,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: &[f32],
        lda: i32,
        b: &[f32],
        ldb: i32,
        beta: f32,
        c: &mut [f32],
        ldc: i32,
    );

    /// SAXPY constant times a vector plus a vector.
    fn saxpy(n: i32, alpha: f32, x: &[f32], incx: i32, y: &mut [f32], incy: i32);

    /// SDOT forms the dot product of two vectors.
    fn sdot(n: i32, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f32;

    /// SCOPY copies a vector, x, to a vector, y.
    fn scopy(n: i32, x: &[f32], incx: i32, y: &mut [f32], incy: i32);

    /// SSCAL scales a vector by a constant.
    fn sscal(n: i32, alpha: f32, x: &mut [f32], incx: i32);
}

pub enum Blas {
    CBlas(CBlas),
}

// TODO add an argument &self to allow to choose between CBlas and CuBlas and the AMD one too.
impl BlasMethods for Blas {
    fn sgemm(
        layout: Layout,
        transa: Transpose,
        transb: Transpose,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: &[f32],
        lda: i32,
        b: &[f32],
        ldb: i32,
        beta: f32,
        c: &mut [f32],
        ldc: i32,
    ) {
        CBlas::sgemm(
            layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        )
    }

    fn sdot(n: i32, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f32 {
        CBlas::sdot(n, x, incx, y, incy)
    }

    fn scopy(n: i32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
        CBlas::scopy(n, x, incx, y, incy)
    }

    fn saxpy(n: i32, alpha: f32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
        CBlas::saxpy(n, alpha, x, incx, y, incy)
    }

    fn sscal(n: i32, alpha: f32, x: &mut [f32], incx: i32) {
        CBlas::sscal(n, alpha, x, incx)
    }
}
