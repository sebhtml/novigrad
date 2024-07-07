
pub fn center_image_in_field_of_view(
    pixels: Vec<usize>,
    old_width: usize,
    old_height: usize,
    new_width: usize,
    new_height: usize,
    default_pixel: usize,
) -> Vec<usize> {
    let mut new_pixels = vec![default_pixel; new_width * new_height];
    let x_translation = (new_width - old_width) / 2;
    let y_translation = (new_height - old_height) / 2;
    for old_x in 0..old_height {
        for old_y in 0..old_width {
            let pixel = pixels[old_y * old_width + old_x];
            let new_x = old_x + x_translation;
            let new_y = old_y + y_translation;
            new_pixels[new_y * new_width + new_x] = pixel;
        }
    }
    new_pixels
}

pub fn center_examples_in_field_of_view(
    examples: Vec<(Vec<usize>, Vec<usize>)>,
    old_width: usize,
    old_height: usize,
    new_width: usize,
    new_height: usize,
    default_pixel: usize,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let examples = examples
        .into_iter()
        .map(|(input, output)| {
            let input = center_image_in_field_of_view(
                input,
                old_width,
                old_height,
                new_width,
                new_height,
                default_pixel,
            );
            let output = center_image_in_field_of_view(
                output,
                old_width,
                old_height,
                new_width,
                new_height,
                default_pixel,
            );
            (input, output)
        })
        .collect::<Vec<_>>();
    examples
}
