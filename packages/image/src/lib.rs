use std::io::Cursor;

use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use image::{ColorType, DynamicImage, ImageBuffer, ImageFormat, ImageReader, Rgba};
use serde::Serialize;
use tiny_skia::{Pixmap, Transform};
use usvg::{Options, Tree};
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
struct DecodedImage {
    data: Vec<u8>,
    width: u32,
    height: u32,
}

#[derive(Serialize)]
struct Dimensions {
    width: u32,
    height: u32,
}

#[wasm_bindgen]
pub fn decode(buf: &[u8]) -> JsValue {
    let dyn_img = image::load_from_memory(buf).expect("failed to decode image");
    let rgba = dyn_img.to_rgba8();
    let out = DecodedImage {
        data: rgba.as_raw().clone(),
        width: rgba.width(),
        height: rgba.height(),
    };
    serde_wasm_bindgen::to_value(&out).expect("failed to serialize decode output")
}

#[wasm_bindgen]
pub fn dimensions(buf: &[u8]) -> JsValue {
    let cursor = Cursor::new(buf);
    let reader = ImageReader::new(cursor)
        .with_guessed_format()
        .expect("failed to guess image format");
    let (width, height) = reader.into_dimensions().expect("failed to read dimensions");
    serde_wasm_bindgen::to_value(&Dimensions { width, height })
        .expect("failed to serialize dimensions output")
}

fn is_svg(buf: &[u8]) -> bool {
    let trimmed = match std::str::from_utf8(buf) {
        Ok(s) => s.trim_start_matches(char::is_whitespace),
        Err(_) => return false,
    };

    trimmed.starts_with("<svg") || trimmed.starts_with("<?xml")
}

#[wasm_bindgen]
pub fn render_svg(svg_data: &[u8], max_width: u32, max_height: u32) -> JsValue {
    let svg = std::str::from_utf8(svg_data).expect("failed to parse SVG as UTF-8");
    let opt = Options::default();
    let tree = Tree::from_str(svg, &opt).expect("failed to parse SVG");

    let size = tree.size();
    let intrinsic_width = size.width();
    let intrinsic_height = size.height();
    assert!(
        intrinsic_width > 0.0 && intrinsic_height > 0.0,
        "SVG must have positive intrinsic size"
    );

    let max_width = if max_width == 0 { 1 } else { max_width };
    let max_height = if max_height == 0 { 1 } else { max_height };

    let scale = (max_width as f32 / intrinsic_width)
        .min(max_height as f32 / intrinsic_height);
    let width = ((intrinsic_width * scale).round() as u32).max(1);
    let height = ((intrinsic_height * scale).round() as u32).max(1);

    let sx = width as f32 / intrinsic_width;
    let sy = height as f32 / intrinsic_height;

    let mut pixmap = Pixmap::new(width, height).expect("failed to allocate pixmap");
    let mut pixmap_mut = pixmap.as_mut();
    resvg::render(&tree, Transform::from_scale(sx, sy), &mut pixmap_mut);

    let out = DecodedImage {
        data: pixmap.data().to_vec(),
        width,
        height,
    };
    serde_wasm_bindgen::to_value(&out).expect("failed to serialize SVG render output")
}

#[wasm_bindgen]
pub fn format(buf: &[u8]) -> String {
    if is_svg(buf) {
        return "svg".to_string();
    }

    let fmt = image::guess_format(buf).expect("unknown image format");
    match fmt {
        ImageFormat::Png => "png",
        ImageFormat::Jpeg => "jpeg",
        ImageFormat::Gif => "gif",
        ImageFormat::Bmp => "bmp",
        ImageFormat::Tiff => "tiff",
        ImageFormat::WebP => "webp",
        _ => "unknown",
    }
    .to_string()
}

#[wasm_bindgen]
pub fn resize(
    rgba: &[u8],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> Vec<u8> {
    let src = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(src_width, src_height, rgba.to_vec())
        .expect("invalid RGBA buffer dimensions");
    let resized = image::imageops::resize(&src, dst_width, dst_height, FilterType::Lanczos3);
    resized.into_raw()
}

#[wasm_bindgen]
pub fn encode_png(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    let img = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(width, height, rgba.to_vec())
        .expect("invalid RGBA buffer dimensions");
    let dyn_img = DynamicImage::ImageRgba8(img);

    let mut out = Vec::new();
    dyn_img
        .write_to(&mut Cursor::new(&mut out), ImageFormat::Png)
        .expect("failed to encode PNG");
    out
}

#[wasm_bindgen]
pub fn encode_jpeg(rgba: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    let img = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(width, height, rgba.to_vec())
        .expect("invalid RGBA buffer dimensions");
    let dyn_img = DynamicImage::ImageRgba8(img);
    let rgb_img = dyn_img.to_rgb8();

    let mut out = Vec::new();
    let q = quality.clamp(1, 100);
    let mut encoder = JpegEncoder::new_with_quality(&mut out, q);
    encoder
        .encode(rgb_img.as_raw(), width, height, ColorType::Rgb8.into())
        .expect("failed to encode JPEG");
    out
}

#[wasm_bindgen]
pub fn pixel_diff(img1: &[u8], img2: &[u8], width: u32, height: u32) -> f64 {
    let expected_len = (width as usize) * (height as usize) * 4;
    assert!(
        img1.len() == expected_len && img2.len() == expected_len,
        "input buffers must match width*height*4"
    );

    let sum_abs_diff: u64 = img1
        .iter()
        .zip(img2.iter())
        .map(|(a, b)| (*a as i16 - *b as i16).unsigned_abs() as u64)
        .sum();

    sum_abs_diff as f64 / (expected_len as f64 * 255.0)
}