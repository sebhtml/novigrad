use crate::graph_theory::partition_branching_graph;

#[test]
fn problem_1() {
    /*
       0
       1
    2 3 4 5
       6
       7
        */
    let edges = vec![
        (0, 1),
        (1, 2),
        (2, 6),
        (1, 3),
        (3, 6),
        (1, 4),
        (4, 6),
        (1, 5),
        (5, 6),
        (6, 7),
    ];

    let vertices = (0..8).collect::<Vec<_>>();

    let paths = partition_branching_graph(&vertices, &edges);

    println!("vertices {:?}", vertices);
    println!("edges {:?}", edges);
    println!("paths: {:?}", paths);

    assert_eq!(6, paths.len());
    assert_eq!(vec![(0, 1)], paths[0]);
    assert_eq!(vec![(1, 2), (2, 6)], paths[1]);
    assert_eq!(vec![(1, 3), (3, 6)], paths[2]);
    assert_eq!(vec![(1, 4), (4, 6)], paths[3]);
    assert_eq!(vec![(1, 5), (5, 6)], paths[4]);
    assert_eq!(vec![(6, 7)], paths[5]);
}

#[test]
fn problem_2() {
    /*
       0
       1
    2 3 4 5
    6 7 8 9
       10
       11
       12
        */
    let edges = vec![
        (0, 1),
        (1, 2),
        (2, 6),
        (6, 10),
        (1, 3),
        (3, 7),
        (7, 10),
        (1, 4),
        (4, 8),
        (8, 10),
        (1, 5),
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),
    ];

    let vertices = (0..13).collect::<Vec<_>>();

    let paths = partition_branching_graph(&vertices, &edges);

    println!("vertices {:?}", vertices);
    println!("edges {:?}", edges);
    println!("paths: {:?}", paths);

    assert_eq!(6, paths.len());
    assert_eq!(vec![(0, 1)], paths[0]);
    assert_eq!(vec![(1, 2), (2, 6), (6, 10)], paths[1]);
    assert_eq!(vec![(1, 3), (3, 7), (7, 10)], paths[2]);
    assert_eq!(vec![(1, 4), (4, 8), (8, 10)], paths[3]);
    assert_eq!(vec![(1, 5), (5, 9), (9, 10)], paths[4]);
    assert_eq!(vec![(10, 11), (11, 12)], paths[5]);
}
