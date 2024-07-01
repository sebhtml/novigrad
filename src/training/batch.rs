use rand::prelude::SliceRandom;
use rand::thread_rng;

pub fn make_batches(
    indices: &[usize],
    shuffle_examples: bool,
    batch_size: usize,
) -> Vec<Vec<usize>> {
    let mut indices = indices.to_owned();
    if shuffle_examples {
        indices.shuffle(&mut thread_rng());
    }

    let mut batches: Vec<Vec<_>> = vec![];
    for index in indices {
        if batches.len() == 0 || batches[batches.len() - 1].len() == batch_size {
            batches.push(vec![]);
        }
        let last = batches.len() - 1;
        let batch = &mut batches[last];
        batch.push(index);
    }
    batches
}
