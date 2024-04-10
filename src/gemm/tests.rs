/// This is the example from https://docs.rs/cblas/latest/cblas/.
#[test]
fn sgemm_column_major() {
    use cblas::*;

    let (m, n, k) = (2, 4, 3);
    let a = vec![
        //
        1.0, 4.0, //
        2.0, 5.0, //
        3.0, 6.0, //
    ];
    let b = vec![
        //
        1.0, 5.0, 9.0, //
        2.0, 6.0, 10.0, //
        3.0, 7.0, 11.0, //
        4.0, 8.0, 12.0, //
    ];
    let mut c = vec![
        //
        2.0, 7.0, //
        6.0, 2.0, //
        0.0, 7.0, //
        4.0, 2.0, //
    ];

    unsafe {
        sgemm(
            Layout::ColumnMajor,
            Transpose::None,
            Transpose::None,
            m,
            n,
            k,
            1.0,
            &a,
            m,
            &b,
            k,
            1.0,
            &mut c,
            m,
        );
    }

    assert_eq!(
        c,
        vec![
            //
            40.0, 90.0, //
            50.0, 100.0, //
            50.0, 120.0, //
            60.0, 130.0, //
        ]
    );
}

#[test]
fn sgemm_row_major() {
    use cblas::*;

    let (m, n, k) = (2, 4, 3);
    let a = vec![
        //
        1.0, 2.0, 3.0, //
        4.0, 5.0, 6.0, //
    ];
    let b = vec![
        //
        1.0, 2.0, 3.0, 4.0, //
        5.0, 6.0, 7.0, 8.0, //
        9.0, 10.0, 11.0, 12.0, //
    ];
    let mut c = vec![
        //
        2.0, 6.0, 0.0, 4.0, //
        7.0, 2.0, 7.0, 2.0, //
    ];

    unsafe {
        sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::None,
            m,
            n,
            k,
            1.0,
            &a,
            k,
            &b,
            n,
            1.0,
            &mut c,
            n,
        );
    }

    assert_eq!(
        c,
        vec![
            //
            40.0, 50.0, 50.0, 60.0, //
            90.0, 100.0, 120.0, 130.0, //
        ]
    );
}

#[test]
fn sgemm_row_major_a_transpose() {
    use cblas::*;

    let (m, n, k) = (2, 4, 3);
    let lda = m;
    let a = vec![
        //
        1.0, 4.0, //
        2.0, 5.0, //
        3.0, 6.0, //
    ];
    let b = vec![
        //
        1.0, 2.0, 3.0, 4.0, //
        5.0, 6.0, 7.0, 8.0, //
        9.0, 10.0, 11.0, 12.0, //
    ];
    let mut c = vec![
        //
        2.0, 6.0, 0.0, 4.0, //
        7.0, 2.0, 7.0, 2.0, //
    ];

    unsafe {
        sgemm(
            Layout::RowMajor,
            Transpose::Ordinary,
            Transpose::None,
            m,
            n,
            k,
            1.0,
            &a,
            lda,
            &b,
            n,
            1.0,
            &mut c,
            n,
        );
    }

    assert_eq!(
        c,
        vec![
            //
            40.0, 50.0, 50.0, 60.0, //
            90.0, 100.0, 120.0, 130.0, //
        ]
    );
}
