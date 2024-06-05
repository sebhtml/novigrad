use crate::graph_theory::partition_branching_graph;

#[test]
fn problem_6_partitions() {
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
fn problem_2_with_4_paths() {
    let edges = vec![
        (0, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 6),
        (3, 7),
        (4, 8),
        (5, 9),
        (6, 10),
        (7, 10),
        (8, 10),
        (9, 10),
    ];

    let vertices = (0..11).collect::<Vec<_>>();

    let paths = partition_branching_graph(&vertices, &edges);

    println!("paths: {:?}", paths);
    assert_eq!(6, paths.len());

}
