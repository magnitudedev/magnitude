use magnitude_chat::{
    BpeConfig, ByteBpeTokenizer, ChatRequest, Event, FinishReason, OutputToken, PieceKind,
    PreparedChat, SpecialTokens, TemplateBundle, TemplateSelection, TemplateVariant, TerminalCause,
    TokenChatStream, TokenId,
};
use std::collections::BTreeSet;

#[test]
fn method_policy_is_qualified_against_the_prepared_method() {
    use magnitude_chat::wire::MethodPolicy;

    assert!(MethodPolicy::Plain.validate_method("plain").is_ok());
    assert!(MethodPolicy::Plain
        .validate_method("mtp:artifact:3")
        .is_err());
    let mtp = MethodPolicy::Mtp {
        greedy_proposals: 2,
        sampled_proposals: 1,
    };
    assert!(mtp.validate_method("mtp:artifact:3").is_ok());
    assert!(mtp.validate_method("mtp:artifact:1").is_err());
    assert!(mtp.validate_method("plain").is_err());
}

fn tokenizer() -> ByteBpeTokenizer {
    let mut bytes: Vec<u8> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut alphabet: Vec<u32> = bytes.iter().map(|&byte| u32::from(byte)).collect();
    let mut next = 256;
    for byte in 0..=255 {
        if !bytes.contains(&byte) {
            bytes.push(byte);
            alphabet.push(next);
            next += 1;
        }
    }
    let mut pieces = vec![String::new(); 256];
    for (byte, code) in bytes.into_iter().zip(alphabet) {
        pieces[byte as usize] = char::from_u32(code).unwrap().to_string();
    }
    pieces.push("<eos>".into());
    let mut kinds = vec![PieceKind::Normal; 256];
    kinds.push(PieceKind::Control);
    ByteBpeTokenizer::new(BpeConfig {
        artifact_identity: "fixture".into(),
        pieces,
        kinds,
        merges: vec![],
        pattern: r".+|\s".into(),
        normalize_nfc: true,
        stop_tokens: BTreeSet::from([TokenId(256)]),
    })
    .unwrap()
}

fn bundle(suffix: &str) -> TemplateBundle {
    TemplateBundle::new(
        vec![TemplateVariant {
            name: "default".into(),
            source: format!("{{{{ messages[0].content }}}}{suffix}"),
            provenance: "fixture".into(),
        }],
        "default".into(),
        Default::default(),
    )
    .unwrap()
}

fn prepared(tokenizer: &ByteBpeTokenizer, suffix: &str) -> PreparedChat {
    let bundle = bundle(suffix);
    PreparedChat::prepare(
        &bundle,
        tokenizer,
        &ChatRequest::new(vec![serde_json::json!({"role":"user", "content":"é"})], 0),
        &TemplateSelection::default(),
    )
    .unwrap()
}

#[test]
fn wire_accepts_and_forwards_all_sampling_shaping_options() {
    use magnitude_chat::wire::{MethodPolicy, ModelLimits, Request};

    let tokenizer = tokenizer();
    let bundle = bundle("");
    let request = Request::parse(
        br#"{"model":"fixture","messages":[{"role":"user","content":"hello"}],"temperature":0.5,"top_p":0.8,"top_k":17,"min_p":0.1,"repetition_penalty":1.2,"presence_penalty":-0.3,"frequency_penalty":0.4}"#,
        1 << 20,
    )
    .unwrap();
    let prepared = request
        .prepare(
            &bundle,
            &tokenizer,
            &TemplateSelection::default(),
            0,
            &ModelLimits {
                model: "fixture",
                context_tokens: 128,
                vocabulary: tokenizer.vocabulary(),
                output_capacity: 8,
                forced_quantum: 4,
                method: MethodPolicy::Plain,
                media_marker: None,
            },
        )
        .unwrap();
    assert_eq!(
        prepared.options.sampling,
        magnitude_chat::Sampling::Categorical
    );
    assert_eq!(prepared.options.shaping.temperature, 0.5);
    assert_eq!(prepared.options.shaping.top_p, 0.8);
    assert_eq!(prepared.options.shaping.top_k, 17);
    assert_eq!(prepared.options.shaping.min_p, 0.1);
    assert_eq!(prepared.options.shaping.repetition_penalty, 1.2);
    assert_eq!(prepared.options.shaping.presence_penalty, -0.3);
    assert_eq!(prepared.options.shaping.frequency_penalty, 0.4);

    for body in [
        br#"{"model":"fixture","messages":[{"role":"user","content":"x"}],"temperature":-0.1}"#.as_slice(),
        br#"{"model":"fixture","messages":[{"role":"user","content":"x"}],"top_p":0}"#.as_slice(),
        br#"{"model":"fixture","messages":[{"role":"user","content":"x"}],"min_p":1.1}"#.as_slice(),
        br#"{"model":"fixture","messages":[{"role":"user","content":"x"}],"repetition_penalty":0}"#.as_slice(),
        br#"{"model":"fixture","messages":[{"role":"user","content":"x"}],"presence_penalty":1e300}"#.as_slice(),
        br#"{"model":"fixture","messages":[{"role":"user","content":"x"}],"top_k":16777217}"#.as_slice(),
        br#"{"model":"fixture","messages":[{"role":"user","content":"x"}],"top_k":4294967296}"#.as_slice(),
    ] {
        assert!(Request::parse(body, 1 << 20).is_err());
    }
}

#[test]
fn tokenizer_preserves_ids_and_incremental_utf8() {
    let tokenizer = tokenizer();
    let encoded = tokenizer
        .encode("café 世界", SpecialTokens::Recognize)
        .unwrap();
    assert_eq!(
        encoded,
        "café 世界"
            .as_bytes()
            .iter()
            .map(|&byte| TokenId(u32::from(byte)))
            .collect::<Vec<_>>()
    );
    let mut decoder = tokenizer.decoder(false);
    let mut decoded = String::new();
    for token in encoded {
        decoded.push_str(&decoder.push(token).unwrap());
    }
    decoded.push_str(&decoder.finish().unwrap());
    assert_eq!(decoded, "café 世界");
}

#[test]
fn preparation_counts_the_exact_rendered_prompt() {
    let tokenizer = tokenizer();
    let prepared = prepared(&tokenizer, "<eos>");
    assert_eq!(prepared.prompt(), "é<eos>");
    assert_eq!(prepared.prompt_tokens(), 3);
    assert_eq!(
        prepared.input().tokens,
        vec![TokenId(195), TokenId(169), TokenId(256)]
    );
    assert_eq!(prepared.input().tokenizer_identity, tokenizer.identity());
}

#[test]
fn token_stream_is_chunk_invariant_and_stops_before_semantic_parsing() {
    let tokenizer = tokenizer();
    let prepared = prepared(&tokenizer, "");
    let mut stream =
        TokenChatStream::new(&prepared, &tokenizer, vec!["世界".into()], 4096).unwrap();
    let mut events = Vec::new();
    for (index, &byte) in "café 世界 ignored".as_bytes().iter().enumerate() {
        events.extend(
            stream
                .feed(&OutputToken {
                    index,
                    token: TokenId(u32::from(byte)),
                })
                .unwrap(),
        );
        if stream.stopped() {
            break;
        }
    }
    let content = events
        .iter()
        .filter_map(|event| match event {
            Event::Content { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert_eq!(content, "café ");
    assert_eq!(
        events.last(),
        Some(&Event::Finish {
            cause: TerminalCause::UserStop,
        })
    );
    assert!(stream.finish(FinishReason::Cancelled).is_err());
}

/// A JSON-schema request starts generating under every forced-run quantum.
/// Quantum 0 (the served engine's setting) disables forced runs: the final
/// prefill chunk selects its first token under the grammar mask.
#[test]
fn json_constrained_generation_starts_with_and_without_forced_runs() {
    use magnitude_artifacts::InputLayout;
    use magnitude_chat::{CacheLimits, Options, Sampling, Vocabulary};
    use magnitude_generation::{Plain, RequestId, RoundStart, Shaping, WorkKind};
    use magnitude_model_contracts::{PreparedModelInput, TokenPlan};
    use std::sync::Arc;

    let tokenizer = Arc::new(tokenizer());
    let mut request = ChatRequest::new(vec![serde_json::json!({"role":"user", "content":"hi"})], 0);
    request.json_schema = Some(serde_json::json!({
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": false
    }));
    let prepared = PreparedChat::prepare(
        &bundle(""),
        &tokenizer,
        &request,
        &TemplateSelection::default(),
    )
    .unwrap();
    let tokens = prepared.input().tokens.clone();
    let coordinates = (0..tokens.len())
        .map(|position| [position as i32; 3])
        .collect();
    let layout = InputLayout::new(tokens.len(), Vec::new()).unwrap();
    let plan = PreparedModelInput::from_text_coordinates(
        TokenPlan::new(tokens.clone(), layout).unwrap(),
        coordinates,
    )
    .unwrap();
    let mut vocabulary = Vocabulary::new(
        tokenizer.clone(),
        tokenizer.vocabulary(),
        CacheLimits {
            entries: 2,
            bytes: 1 << 20,
        },
    )
    .unwrap();
    for forced_quantum in [0, 4] {
        let options = Options {
            max_tokens: 64,
            output_capacity: 16,
            context_limit: 256,
            vocabulary: tokenizer.vocabulary(),
            stop_tokens: tokenizer.stop_tokens().clone(),
            sampling: Sampling::Greedy,
            shaping: Shaping {
                temperature: 0.0,
                ..Default::default()
            },
            seed: 0,
            forced_quantum,
            method: magnitude_generation::MethodChoice::Plain,
        };
        let mut generation = vocabulary
            .prepare_generation_for_input(prepared.input(), options, &plan)
            .unwrap()
            .into_generation(Arc::new(Plain))
            .unwrap();
        assert_eq!(
            generation.start_round(RequestId(1), 64).unwrap(),
            RoundStart::Target
        );
        let forward = generation.round_forward().unwrap();
        assert_eq!(forward.kind, WorkKind::Prefill);
        assert_eq!(forward.tokens, tokens);
        // The schema admits leading whitespace, so no token is forced: the
        // first output is selected under the grammar mask.
        assert_eq!(forward.selects.len(), 1);
        assert!(forward.selects[0].mask.is_some());
    }
}
