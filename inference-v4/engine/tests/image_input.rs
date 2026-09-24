//! Image requests through the composition root on the pinned Qwen3.5 4B and
//! its projector: the chat wire renders each image part as the Qwen
//! placeholder, and input preparation expands it to the image's merged patch
//! rows between the vision delimiters, one conditioned span per image, with
//! the 2D rotary coordinates of the image and the text after it.
//!
//! `cargo test --release -p magnitude-engine --test image_input -- --ignored`
//! (`VISION_IMAGE` overrides the image, default llama.cpp's `test-1.jpeg`).

use magnitude_engine::{
    chat::{
        wire::{MethodPolicy, ModelLimits, Request},
        TemplateSelection,
    },
    composition::{LoadedArtifacts, MediaSourcePolicy},
};
use serde_json::json;
use std::path::PathBuf;

const PINNED_4B: [&str; 2] = [
    "/.cache/huggingface/hub/models--unsloth--Qwen3.5-4B-GGUF/snapshots/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf",
    "/models/unsloth-qwen3.5-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf",
];
const PROJECTOR: &str = "/models/qwen3.5-4b-vision/mmproj-F16.gguf";
const PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";

fn home(relative: &str) -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap() + relative)
}

fn artifacts() -> LoadedArtifacts {
    let target = PINNED_4B
        .iter()
        .map(|relative| home(relative))
        .find(|path| path.exists())
        .expect("the pinned 4B GGUF");
    LoadedArtifacts::open_with_projector(target, home(PROJECTOR)).unwrap()
}

fn image_url() -> String {
    let path = std::env::var("VISION_IMAGE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| home("/repos/llama.cpp/tools/mtmd/test-1.jpeg"));
    format!("data:image/jpeg;base64,{}", base64::encode(std::fs::read(path).unwrap()))
}

fn prepare(artifacts: &LoadedArtifacts, images: usize) -> magnitude_model_contracts::PreparedModelInput {
    let mut content = vec![json!({"type":"text","text":"Compare:"})];
    for _ in 0..images {
        content.push(json!({"type":"image_url","image_url":{"url": image_url()}}));
    }
    content.push(json!({"type":"text","text":"Describe it."}));
    let body = json!({
        "model": "Qwen3.5-4B-Q4_K_M",
        "messages": [{"role":"user","content": content}],
        "chat_template_kwargs": {"enable_thinking": false},
    });
    let request = Request::parse(&serde_json::to_vec(&body).unwrap(), 64 << 20).unwrap();
    let limits = ModelLimits {
        model: "Qwen3.5-4B-Q4_K_M",
        context_tokens: 8192,
        vocabulary: artifacts.tokenizer().vocabulary(),
        output_capacity: 64,
        forced_quantum: 0,
        method: MethodPolicy::Plain,
        media_marker: Some(PLACEHOLDER),
    };
    let prepared = request
        .prepare(artifacts.templates(), artifacts.tokenizer(), &TemplateSelection::default(), 0, &limits)
        .unwrap();
    assert_eq!(prepared.image_sources.len(), images);
    let policy = MediaSourcePolicy::data_urls_only();
    artifacts.prepare_input(&prepared, |source| policy.resolve(source)).unwrap()
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
    let (start, pad, end) = (token("<|vision_start|>"), token("<|image_pad|>"), token("<|vision_end|>"));

    let input = prepare(&artifacts, 1);
    let [span] = input.layout().spans() else { panic!("one image span") };
    let [vision] = input.vision() else { panic!("one vision input") };
    assert_eq!(span.identity, vision.identity());
    let [t, h, w] = vision.grid();
    assert_eq!(t, 1, "a still image is one temporal patch");
    assert_eq!(span.end - span.start, h * w / 4, "one row per merged 2 x 2 patch block");
    let tokens = input.tokens();
    assert_eq!(tokens[span.start - 1], start);
    assert!(tokens[span.start..span.end].iter().all(|token| *token == pad));
    assert_eq!(tokens[span.end], end);
    assert_eq!(tokens.iter().filter(|token| **token == pad).count(), span.end - span.start);

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
    let [first, second] = two.layout().spans() else { panic!("two image spans") };
    assert_eq!(first.identity, span.identity);
    assert_eq!(second.identity, span.identity);
    assert_eq!(second.end - second.start, span.end - span.start);
    // The second image starts past the first image's extent.
    let first_after = two.coordinates()[first.end][0];
    assert_eq!(two.coordinates()[second.start][0], first_after + (second.start - first.end) as i32);
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
