#[cfg(test)]
mod tests;

#[allow(unused)]
pub fn find_vertex_disjoint_paths(vertices: &[usize], edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
    println!("vertices {:?}", vertices);
    println!("edges {:?}", edges);
    let mut dependencies = vec![Vec::<usize>::default(); vertices.len()];
    let mut dependents = vec![Vec::<usize>::default(); vertices.len()];
    for (dependency, dependent) in edges.iter() {
        dependencies[*dependent].push(*dependency);
        dependents[*dependency].push(*dependent);
    }
    let mut paths = vec![];
    let mut in_path = vec![false; vertices.len()];
    for v1 in 0..vertices.len() {
        for v2 in 0..vertices.len() {
            if v1 == v2 {
                continue;
            }

            let v1_dependents = &dependents[v1];
            let v2_dependencies = &dependencies[v2];

            if v1_dependents.len() >= 2 && v1_dependents.len() == v2_dependencies.len() {
                for dependent in v1_dependents.iter() {
                    paths.push(vec![*dependent]);
                    in_path[*dependent] = true;
                }
            }
        }
    }

    for (dependency, dependent) in edges.iter() {
        if !in_path[*dependency] && !in_path[*dependent] {
            paths.push(vec![*dependency, *dependent]);
            in_path[*dependency] = true;
            in_path[*dependent] = true;
        }
    }

    for vertex in vertices.iter() {
        if !in_path[*vertex] {
            paths.push(vec![*vertex]);
            in_path[*vertex] = true;
        }
    }
    paths
}
