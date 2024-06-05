#[cfg(test)]
mod tests;

#[allow(unused)]
pub fn partition_branching_graph(
    vertices: &[usize],
    edges: &[(usize, usize)],
) -> Vec<Vec<(usize, usize)>> {
    let mut dependencies = vec![Vec::<usize>::default(); vertices.len()];
    let mut dependents = vec![Vec::<usize>::default(); vertices.len()];
    for (dependency, dependent) in edges.iter() {
        dependencies[*dependent].push(*dependency);
        dependents[*dependency].push(*dependent);
    }

    // Step 1: Find all pairs of vertices (u, v)
    let mut uv_pairs = vec![];

    for u in 0..vertices.len() {
        for v in 0..vertices.len() {
            if u == v {
                continue;
            }

            let u_dependents = &dependents[u];
            let v_dependencies = &dependencies[v];

            if u_dependents.len() >= 2 && u_dependents.len() == v_dependencies.len() {
                uv_pairs.push(((u, v), u_dependents.len()));
            }
        }
    }

    let mut partitions = vec![];

    // For each pair (u, v)
    for ((u, v), n) in uv_pairs.iter() {
        // With pair (u1, v1) as the next pair, if any
        // Get { all ascendents of u } ∪ { u }
        // For each w in dependants of u
        //   With z as the dependency of v that is a descendent of w
        //   Get { w } ∪ ( { all descendents of w } ∩ { all ascendents of z } ) ∪ { z }
        // Get { v } ∪ ( { all descendents of v } ∩ { all ascendents of u2 } )

        let edge_u = edges
            .iter()
            .enumerate()
            .find(|(i, (a, b))| b == u)
            .map(|(i, _)| i)
            .unwrap();
        let edge_v = edges
            .iter()
            .enumerate()
            .find(|(i, (a, b))| a == v)
            .map(|(i, _)| i)
            .unwrap();

        println!("(u, v) is ({}, {})", u, v);
        println!("edge_u {}", edge_u);
        println!("edge_v {}", edge_v);

        let part_u = edges[0..(edge_u + 1)].to_owned();
        partitions.push(part_u);
        let uv_parts_begin = edge_u + 1;
        let uv_parts_end = edge_v;
        let uv_edges = uv_parts_end - uv_parts_begin;
        let uv_edges_per_uv_part = uv_edges / n;
        for i in 0..*n {
            let part_uv_begin = uv_parts_begin + i * uv_edges_per_uv_part;
            let part_uv_end = part_uv_begin + uv_edges_per_uv_part;
            let part_uv_i = edges[part_uv_begin..part_uv_end].to_owned();
            partitions.push(part_uv_i);
        }
        let part_v = edges[edge_v..edges.len()].to_owned();
        partitions.push(part_v);
    }

    partitions
}
