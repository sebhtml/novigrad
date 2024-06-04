use crate::graph_theory::find_vertex_disjoint_paths;

#[test]
fn problem_4_paths() {
    let edges = vec![
        (0, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 6),
        (3, 6),
        (4, 6),
        (5, 6),
        (6, 7),
    ];

    let vertices = (0..8).collect::<Vec<_>>();

    let paths = find_vertex_disjoint_paths(&vertices, &edges);

    println!("paths: {:?}", paths);
    assert_eq!(6, paths.len());
    assert_eq!(vec![2], paths[0]);
    assert_eq!(vec![3], paths[1]);
    assert_eq!(vec![4], paths[2]);
    assert_eq!(vec![5], paths[3]);
    assert_eq!(vec![0, 1], paths[4]);
    assert_eq!(vec![6, 7], paths[5]);
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

    let paths = find_vertex_disjoint_paths(&vertices, &edges);

    println!("paths: {:?}", paths);
    assert_eq!(6, paths.len());
    assert_eq!(vec![2], paths[0]);
    assert_eq!(vec![3], paths[1]);
    assert_eq!(vec![4], paths[2]);
    assert_eq!(vec![5], paths[3]);
    assert_eq!(vec![0, 1], paths[4]);
    assert_eq!(vec![6, 7], paths[5]);
}
