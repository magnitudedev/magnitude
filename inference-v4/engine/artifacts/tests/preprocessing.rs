use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use magnitude_artifacts::{ImageProcessor, ImageProcessorConfig};
use std::io::Cursor;

fn processor(patch: usize, merge: usize, min_pixels: usize, max_pixels: usize) -> ImageProcessor {
    ImageProcessor::new(ImageProcessorConfig {
        kind: "fixture".into(),
        patch,
        merge,
        temporal_patch: 2,
        min_pixels,
        max_pixels,
        mean: [0.0; 3],
        std: [1.0; 3],
    })
    .unwrap()
}

#[test]
fn smart_resize_uses_half_to_even_and_pixel_bounds() {
    let value = processor(16, 2, 65_536, 16_777_216);
    assert_eq!(value.smart_resize(48, 80).unwrap(), (224, 352));
    assert_eq!(value.smart_resize(80, 80).unwrap(), (256, 256));
    assert!(value.smart_resize(1, 201).is_err());

    let downscale = processor(16, 2, 1, 32 * 32);
    assert_eq!(downscale.smart_resize(128, 64).unwrap(), (32, 32));
}

#[test]
fn decode_normalize_and_patchify_follow_merged_block_order() {
    let mut source = RgbImage::new(4, 4);
    for y in 0..4 {
        for x in 0..4 {
            source.put_pixel(x, y, Rgb([(y * 4 + x) as u8, 64, 255]));
        }
    }
    let mut encoded = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(source)
        .write_to(&mut encoded, ImageFormat::Png)
        .unwrap();

    let prepared = processor(1, 2, 16, 16)
        .prepare(&[encoded.into_inner()])
        .unwrap();
    assert_eq!(prepared.processor().len(), 64);
    let values = prepared
        .tensors()
        .iter()
        .find(|tensor| tensor.name() == "pixel_values")
        .unwrap();
    assert_eq!(values.shape(), [16, 6]);
    let floats = values
        .data()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    let first_channel = floats.chunks_exact(6).map(|row| row[0]).collect::<Vec<_>>();
    let expected_order =
        [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15].map(|value| value as f32 / 255.0);
    assert_eq!(first_channel, expected_order);
    assert!(floats
        .chunks_exact(6)
        .all(|row| row[0] == row[1] && row[2] == row[3] && row[4] == row[5]));

    let grid = prepared
        .tensors()
        .iter()
        .find(|tensor| tensor.name() == "image_grid_thw")
        .unwrap();
    assert_eq!(grid.shape(), [1, 3]);
    assert_eq!(
        grid.data()
            .chunks_exact(8)
            .map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>(),
        vec![1, 4, 4]
    );
}

#[test]
fn processor_identity_covers_every_numeric_constant() {
    let base = processor(1, 2, 16, 16);
    let changed = processor(1, 2, 16, 32);
    assert_ne!(base.identity(), changed.identity());
    assert_eq!(base.identity(), processor(1, 2, 16, 16).identity());
}

#[test]
fn bicubic_resize_preserves_pil_rgb_quantization() {
    let mut source = RgbImage::new(3, 3);
    for y in 0..3 {
        for x in 0..3 {
            source.put_pixel(x, y, Rgb([(x * 71 + y * 13) as u8, 97, 203]));
        }
    }
    let mut encoded = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(source)
        .write_to(&mut encoded, ImageFormat::Png)
        .unwrap();
    let prepared = processor(1, 2, 16, 16)
        .prepare(&[encoded.into_inner()])
        .unwrap();
    let pixels = prepared
        .tensors()
        .iter()
        .find(|tensor| tensor.name() == "pixel_values")
        .unwrap();
    for value in pixels
        .data()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
    {
        assert!((value * 255.0 - (value * 255.0).round()).abs() <= 2.0e-5);
    }
}
