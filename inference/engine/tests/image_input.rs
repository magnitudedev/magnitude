//! Image requests through the composition root on the pinned Qwen3.5 4B and
//! its projector: the chat wire renders each image part as the Qwen
//! placeholder, and input preparation expands it to the image's merged patch
//! rows between the vision delimiters, one conditioned span per image, with
//! the 2D rotary coordinates of the image and the text after it.
//!
//! `cargo test --release -p magnitude-engine --test image_input -- --ignored`
//! (`VISION_IMAGE` overrides the image, default llama.cpp's `test-1.jpeg`).

use magnitude_chat::request::ImageInput;
use magnitude_engine::{
    host::HostArtifacts,
    options::{PackageOptions, ProjectorSelection},
};
use std::path::PathBuf;

const PINNED_4B: [&str; 2] = [
    "/.cache/huggingface/hub/models--unsloth--Qwen3.5-4B-GGUF/snapshots/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf",
    "/models/unsloth-qwen3.5-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf",
];
const PROJECTOR: &str = "/models/qwen3.5-4b-vision/mmproj-F16.gguf";

fn home(relative: &str) -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap() + relative)
}

fn artifacts() -> HostArtifacts {
    let target = PINNED_4B
        .iter()
        .map(|relative| home(relative))
        .find(|path| path.exists())
        .expect("the pinned 4B GGUF");
    HostArtifacts::open(&PackageOptions {
        target,
        projector: ProjectorSelection::Explicit(home(PROJECTOR)),
        draft: None,
    })
    .unwrap()
}

fn image() -> ImageInput {
    let path = std::env::var("VISION_IMAGE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| home("/repos/llama.cpp/tools/mtmd/test-1.jpeg"));
    ImageInput {
        media_type: "image/jpeg".into(),
        bytes: std::fs::read(path).unwrap().into(),
    }
}

/// A rendered prompt with one family media placeholder per image, as the
/// chat template renders image parts.
fn prepare(artifacts: &HostArtifacts, images: usize) -> magnitude_family_contracts::PreparedModelInput {
    let placeholder = artifacts.media_placeholder().expect("a vision model");
    let prompt = format!("Compare:{}Describe it.", placeholder.repeat(images));
    let tokens = artifacts
        .tokenizer()
        .encode(&prompt, magnitude_engine::inputs::SpecialTokens::Recognize)
        .unwrap();
    artifacts
        .prepare_input(tokens, &vec![image(); images])
        .unwrap()
}

#[test]
#[ignore]
fn image_parts_expand_to_conditioned_spans_with_spatial_coordinates() {
    let artifacts = artifacts();
    let token = |text: &str| {
        let tokens = artifacts
            .tokenizer()
            .encode(text, magnitude_engine::inputs::SpecialTokens::Recognize)
            .unwrap();
        assert_eq!(tokens.len(), 1, "{text} is one token");
        tokens[0]
    };
    let (start, pad, end) = (
        token("<|vision_start|>"),
        token("<|image_pad|>"),
        token("<|vision_end|>"),
    );

    let input = prepare(&artifacts, 1);
    let [span] = input.layout().spans() else {
        panic!("one image span")
    };
    assert_eq!(input.vision().len(), 1, "one vision input");
    let vision = &input.vision()[&span.identity];
    let [t, h, w] = vision.grid();
    assert_eq!(t, 1, "a still image is one temporal patch");
    assert_eq!(
        span.end - span.start,
        h * w / 4,
        "one row per merged 2 x 2 patch block"
    );
    let tokens = input.tokens();
    assert_eq!(tokens[span.start - 1], start);
    assert!(tokens[span.start..span.end]
        .iter()
        .all(|token| *token == pad));
    assert_eq!(tokens[span.end], end);
    assert_eq!(
        tokens.iter().filter(|token| **token == pad).count(),
        span.end - span.start
    );

    // Text before the image counts positions; the image rows share the
    // temporal coordinate and spread over rows and columns; the text after it
    // resumes past the image's larger side.
    let coordinates = input.coordinates();
    let base = span.start as i32;
    assert_eq!(coordinates[span.start - 1], [base - 1; 3]);
    let (rows, columns) = (h / 2, w / 2);
    for row in 0..rows {
        for column in 0..columns {
            assert_eq!(
                coordinates[span.start + row * columns + column],
                [base, base + row as i32, base + column as i32]
            );
        }
    }
    let after = base + rows.max(columns) as i32;
    assert_eq!(coordinates[span.end], [after; 3]);
    assert!(input.layout().boundary(span.start) && input.layout().boundary(span.end));

    // Preparation is deterministic: the same image has the same identity,
    // twice in one request as well as across requests.
    let again = prepare(&artifacts, 1);
    assert_eq!(again.tokens(), input.tokens());
    assert_eq!(again.layout().spans()[0].identity, span.identity);
    let two = prepare(&artifacts, 2);
    let [first, second] = two.layout().spans() else {
        panic!("two image spans")
    };
    assert_eq!(first.identity, span.identity);
    assert_eq!(second.identity, span.identity);
    assert_eq!(two.vision().len(), 1, "a repeated image is prepared once");
    assert_eq!(second.end - second.start, span.end - span.start);
    // The second image starts past the first image's extent.
    let first_after = two.coordinates()[first.end][0];
    assert_eq!(
        two.coordinates()[second.start][0],
        first_after + (second.start - first.end) as i32
    );
}

#[test]
#[ignore]
fn text_requests_have_no_spans_and_linear_coordinates() {
    let artifacts = artifacts();
    let input = prepare(&artifacts, 0);
    assert!(input.layout().spans().is_empty() && input.vision().is_empty());
    for (position, coordinates) in input.coordinates().iter().enumerate() {
        assert_eq!(*coordinates, [position as i32; 3]);
    }
}
