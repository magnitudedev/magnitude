//! Image input through chat preparation: each image renders as the family's
//! media marker, which the template authors, so the prompt encodes it as the
//! marker's tokens, while caller text spelling the marker stays literal.
mod support;

use magnitude_chat::{
    request::{Conversation, Entry, ImageInput, OutputFormat, Tools, UserPart},
    BpeConfig, ByteBpeTokenizer, ChatInput, Normalization, PieceEncoding, PieceKind, PreparedChat,
    ReasoningIntent, SpecialTokens, Split, SplitBehavior, TemplateSelection, TokenId,
};
use std::collections::BTreeSet;

const QWEN35: &str = include_str!("../../templates/tests/assets/Qwen3.5-4B.jinja");
const LFM: &str = include_str!("../../templates/tests/assets/LFM2.5-8B-A1B.jinja");
/// The Qwen3.5 and LFM3-VL media markers.
const QWEN35_MARKER: &str = "<|vision_start|><|image_pad|><|vision_end|>";
const LFM3_MARKER: &str = "<|vision_start|><|image|><|vision_end|>";
const CONTROLS: [&str; 7] = [
    "<|startoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|image|>",
];

/// Byte pieces 0..=255, then [`CONTROLS`] as control tokens 256.., `<|im_end|>`
/// stopping.
fn tokenizer() -> ByteBpeTokenizer {
    let mut pieces = support::byte_pieces();
    pieces.extend(CONTROLS.map(String::from));
    let mut kinds = vec![PieceKind::Normal; 256];
    kinds.extend([PieceKind::Control; CONTROLS.len()]);
    ByteBpeTokenizer::new(BpeConfig {
        artifact_identity: "fixture".into(),
        pieces,
        kinds,
        merges: vec![],
        normalization: Normalization::None,
        splits: vec![Split {
            pattern: r".+|\s".into(),
            behavior: SplitBehavior::Isolated,
        }],
        encoding: PieceEncoding::ByteLevel,
        ignore_merges: false,
        implicit_bos: None,
        stop_tokens: BTreeSet::from([TokenId(258)]),
        suppressed_tokens: BTreeSet::new(),
    })
    .unwrap()
}

/// How often `needle` occurs in `tokens`.
fn count(tokens: &[TokenId], needle: &[TokenId]) -> usize {
    tokens
        .windows(needle.len())
        .filter(|window| *window == needle)
        .count()
}

/// A user turn of text spelling `marker`, an image, more text and another
/// image, prepared as the engine prepares chat.
fn assert_markers_encode_as_their_tokens(source: &str, bos: &str, marker: &str) {
    let tokenizer = tokenizer();
    let image = ImageInput {
        media_type: "image/png".into(),
        bytes: Vec::new().into(),
    };
    let spelled = format!("see {marker}");
    let input = ChatInput {
        conversation: Conversation::new(
            None,
            vec![Entry::User(vec![
                UserPart::Text(spelled.clone()),
                UserPart::Image(image.clone()),
                UserPart::Text("and".into()),
                UserPart::Image(image),
            ])],
        )
        .unwrap(),
        tools: Tools::none(),
        reasoning: ReasoningIntent::ModelDefault,
        output: OutputFormat::Text,
        template_arguments: Default::default(),
    };
    let rendered = input.render(0, Some(marker)).unwrap();
    assert_eq!(rendered.images.len(), 2);
    let prepared = PreparedChat::prepare(
        &support::bundle(source, bos, "<|im_end|>"),
        &tokenizer,
        &rendered.request,
        &TemplateSelection::default(),
    )
    .unwrap();
    assert!(prepared
        .prompt()
        .contains(&format!("{spelled}{marker}and{marker}")));
    let tokens = &prepared.input().tokens;
    let marker_tokens = tokenizer.encode(marker, SpecialTokens::Recognize).unwrap();
    assert_eq!(marker_tokens.len(), 3, "{marker} is three control tokens");
    assert_eq!(count(tokens, &marker_tokens), 2, "{marker}: one per image");
    let literal = tokenizer.encode(&spelled, SpecialTokens::Literal).unwrap();
    assert!(literal.iter().all(|token| token.0 < 256));
    assert_eq!(count(tokens, &literal), 1, "the caller's spelling is bytes");
}

#[test]
fn qwen35_image_markers_encode_as_marker_tokens() {
    assert_markers_encode_as_their_tokens(QWEN35, "", QWEN35_MARKER);
}

#[test]
fn lfm3_image_markers_encode_as_marker_tokens() {
    assert_markers_encode_as_their_tokens(LFM, "<|startoftext|>", LFM3_MARKER);
}
