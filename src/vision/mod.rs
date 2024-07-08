fn center_image_in_field_of_view(
    image: Vec<usize>,
    old_width: usize,
    old_height: usize,
    new_width: usize,
    new_height: usize,
    default_pixel: usize,
) -> Vec<usize> {
    let mut new_image = vec![default_pixel; new_width * new_height];
    let x_translation = (new_width - old_width) / 2;
    let y_translation = (new_height - old_height) / 2;
    for old_x in 0..old_height {
        for old_y in 0..old_width {
            let pixel = image[old_y * old_width + old_x];
            let new_x = old_x + x_translation;
            let new_y = old_y + y_translation;
            new_image[new_y * new_width + new_x] = pixel;
        }
    }
    new_image
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

fn translate_image_in_field_of_view(
    image: &Vec<usize>,
    width: usize,
    height: usize,
    default_pixel: usize,
    x_translation: i32,
    y_translation: i32,
) -> Vec<usize> {
    let mut new_image = vec![default_pixel; width * height];
    for old_x in 0..height {
        for old_y in 0..width {
            let pixel = image[old_y * width + old_x];
            let new_x = old_x as i32 + x_translation;
            let new_y = old_y as i32 + y_translation;
            if new_x >= 0 && new_x < width as i32 && new_y >= 0 && new_y < height as i32 {
                let new_x = new_x as usize;
                let new_y = new_y as usize;
                new_image[new_y * width + new_x] = pixel;
            }
        }
    }
    new_image
}

pub fn translate_examples_in_field_of_view(
    examples: Vec<(Vec<usize>, Vec<usize>)>,
    width: usize,
    height: usize,
    default_pixel: usize,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let min_translation_x = -3; //-(width as i32) / 3;
    let max_translation_x = 3; //(width / 3) as i32;
    let min_translation_y = -3; //-(height as i32) / 3;
    let max_translation_y = 3; //(height / 3) as i32;

    let mut new_examples = vec![];

    for (input, output) in examples.iter() {
        for translation_x in min_translation_x..(max_translation_x + 1) {
            for translation_y in min_translation_y..(max_translation_y + 1) {
                let input = translate_image_in_field_of_view(
                    input,
                    width,
                    height,
                    default_pixel,
                    translation_x,
                    translation_y,
                );
                let output = translate_image_in_field_of_view(
                    output,
                    width,
                    height,
                    default_pixel,
                    translation_x,
                    translation_y,
                );
                new_examples.push((input, output));
            }
        }
    }

    new_examples
}
