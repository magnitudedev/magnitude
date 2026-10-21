//! Real-device qualification of a projector's image features against an
//! independent encoder's for the same image: llama.cpp's mtmd, dumped with
//! `MTMD_DEBUG_EMBEDDINGS=<path>` (`[i32 rows][i32 width]`, then the F32
//! rows).
//!
//! `MAGNITUDE_TEST_GGUF=M.gguf MAGNITUDE_TEST_MMPROJ=P.gguf VISION_IMAGE=I.png
//! VISION_REFERENCE=R.bin [VISION_TOLERANCE=0.05] [MAGNITUDE_TEST_DEVICE=vulkan]
//! cargo test --release -p magnitude-engine --test image_features -- --ignored
//! --nocapture`. Both encoders must see the same pixels, so the image must be
//! one neither processor resizes. The target only accepts the projector:
//! features do not depend on it.

use magnitude_chat::{request::ImageInput, SpecialTokens};
use magnitude_engine::{
    build_native_domain,
    composition::{EngineConfiguration, ResolvedEngineConfiguration},
    options::{ModelMethod, ModelPolicy, PackageOptions, ProjectorSelection},
};
use magnitude_executor::{
    platform::{DeviceRequest, MemoryReserves},
    ExecutionPath, FeatureReader, FeatureSpan, Outcome, RequestId, ReservedResources,
};
use magnitude_scheduler::ServiceLimits;
use magnitude_state::KvCodec;
use std::path::PathBuf;

fn variable(name: &str) -> PathBuf {
    PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("set {name}")))
}

fn resolved() -> ResolvedEngineConfiguration {
    EngineConfiguration {
        package: PackageOptions {
            target: variable("MAGNITUDE_TEST_GGUF"),
            projector: ProjectorSelection::Explicit(variable("MAGNITUDE_TEST_MMPROJ")),
            draft: None,
        },
        model: ModelPolicy {
            draft_policy: None,
            method: ModelMethod::Plain,
            mtp_proposals: None,
            kv_codec: KvCodec::AffineK8V4,
            lookahead: false,
            exported_logits_rows: 512,
            numerical_profile: None,
        },
        context_tokens: Some(4096),
        service: ServiceLimits {
            prefill_tokens: 512,
            decode_tokens: 32,
            decode_share: 0.5,
            locality_seconds: 1.0,
        },
        path: ExecutionPath::Native,
        device: std::env::var("MAGNITUDE_TEST_DEVICE")
            .map(|device| device.parse().unwrap())
            .unwrap_or(DeviceRequest::Automatic),
        kernel_cache: std::env::var_os("MAGNITUDE_TEST_KERNEL_CACHE").map(PathBuf::from),
        reserves: MemoryReserves::standard(),
    }
    .resolve()
    .unwrap()
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

/// The reference dump's rows and width, and its values.
fn reference() -> (usize, usize, Vec<f32>) {
    let bytes = std::fs::read(variable("VISION_REFERENCE")).unwrap();
    let header = |at: usize| i32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
    let (rows, width) = (header(0), header(4));
    let values = floats(&bytes[8..]);
    assert_eq!(values.len(), rows * width, "reference dump size");
    (rows, width, values)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot = |x: &[f32], y: &[f32]| {
        x.iter()
            .zip(y)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum::<f64>()
    };
    dot(a, b) / (dot(a, a) * dot(b, b)).sqrt()
}

#[test]
#[ignore = "requires a Metal or CUDA device, MAGNITUDE_TEST_GGUF, MAGNITUDE_TEST_MMPROJ, VISION_IMAGE and VISION_REFERENCE"]
fn image_features_match_the_reference_encoder() {
    let resolved = resolved();
    let host = &resolved.host;
    let placeholder = host.media_placeholder().expect("a model with a projector");
    let image_path = variable("VISION_IMAGE");
    let image = ImageInput {
        media_type: match image_path
            .extension()
            .and_then(|extension| extension.to_str())
        {
            Some("png") => "image/png",
            _ => "image/jpeg",
        }
        .into(),
        bytes: std::fs::read(&image_path).unwrap().into(),
    };
    let tokens = host
        .tokenizer()
        .encode(placeholder, SpecialTokens::Recognize)
        .unwrap();
    let input = host.prepare_input(tokens, &[image]).unwrap();
    let [span] = input.layout().spans() else {
        panic!("one image span")
    };
    let cells = span.end - span.start;

    let (mut domain, mut bindings, _) =
        build_native_domain(&resolved.manifest, host.shared_package()).unwrap();
    let request = RequestId(1);
    domain.install_input(request, input).unwrap();
    let encodes = domain.open_state(&mut bindings, request, None).unwrap();
    let [operation] = encodes.as_slice() else {
        panic!("one encode")
    };
    let ReservedResources::Vision(workspace, output) = domain
        .reserve(&mut bindings, std::slice::from_ref(operation))
        .unwrap()
        .into_resources()
    else {
        panic!("an encode runs on the vision lane")
    };
    let started = std::time::Instant::now();
    let flight = domain
        .submit_vision(bindings, operation, workspace, output)
        .unwrap_or_else(|failure| panic!("{}", failure.error()));
    let (pending, _) = domain.finish_vision(flight).unwrap();
    let encoded = started.elapsed();
    let Outcome::Encode { features } = pending.outcome() else {
        panic!("an encode outcome")
    };
    let rows = domain
        .read(&FeatureSpan::new(features.clone(), 0, cells).unwrap())
        .unwrap();
    let width = rows.row_bytes() / 4;
    let ours = floats(rows.bytes());

    let (reference_rows, reference_width, theirs) = reference();
    assert_eq!(
        (cells, width),
        (reference_rows, reference_width),
        "feature geometry"
    );
    assert!(
        ours.iter().all(|value| value.is_finite()),
        "non-finite features"
    );
    let difference = ours
        .iter()
        .zip(&theirs)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
        .sum::<f64>();
    let magnitude = theirs.iter().map(|b| f64::from(*b).powi(2)).sum::<f64>();
    let relative = (difference / magnitude).sqrt();
    let cosines = ours
        .chunks_exact(width)
        .zip(theirs.chunks_exact(width))
        .map(|(a, b)| cosine(a, b))
        .collect::<Vec<_>>();
    let worst = cosines.iter().copied().fold(f64::INFINITY, f64::min);
    let mean = cosines.iter().sum::<f64>() / cosines.len() as f64;
    let largest = |values: &[f32]| values.iter().fold(0f32, |m, v| m.max(v.abs()));
    eprintln!(
        "image features: {cells} x {width}, encoded in {:.1} ms; relative RMS error {relative:.5}, \
         row cosine mean {mean:.6} min {worst:.6}, max |ours| {} |reference| {}",
        encoded.as_secs_f64() * 1e3,
        largest(&ours),
        largest(&theirs),
    );
    let tolerance = std::env::var("VISION_TOLERANCE")
        .map(|value| value.parse::<f64>().unwrap())
        .unwrap_or(0.05);
    assert!(
        relative < tolerance,
        "relative RMS error {relative} exceeds {tolerance}"
    );
}
