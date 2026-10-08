mod support;

use magnitude_chat::{
    generation::{generation_options, MethodPolicy, ModelLimits},
    request::{PromptCache, SamplingControls},
    BpeConfig, ByteBpeTokenizer, ChatError, ChatRequest, EndOfGeneration, Event, FinishReason,
    GenerationControls, Normalization, OutputToken, PieceEncoding, PieceKind, PreparedChat,
    SpecialTokens, Split, SplitBehavior, TemplateBundle, TemplateSelection, TemplateVariant,
    TerminalCause, TokenChatStream, TokenId,
};
use std::collections::BTreeSet;

#[test]
fn method_policy_is_qualified_against_the_prepared_method() {
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

/// Byte pieces 0..=255, `<eos>` 256 and `<bos>` 257.
fn config() -> BpeConfig {
    let mut pieces = support::byte_pieces();
    pieces.extend(["<eos>".into(), "<bos>".into()]);
    let mut kinds = vec![PieceKind::Normal; 256];
    kinds.extend([PieceKind::Control, PieceKind::Control]);
    BpeConfig {
        artifact_identity: "fixture".into(),
        pieces,
        kinds,
        merges: vec![],
        normalization: Normalization::Nfc,
        splits: vec![Split {
            pattern: r".+|\s".into(),
            behavior: SplitBehavior::Isolated,
        }],
        encoding: PieceEncoding::ByteLevel,
        ignore_merges: false,
        implicit_bos: None,
        stop_tokens: BTreeSet::from([TokenId(256)]),
        suppressed_tokens: BTreeSet::new(),
    }
}

fn tokenizer() -> ByteBpeTokenizer {
    ByteBpeTokenizer::new(config()).unwrap()
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

fn controls(sampling: SamplingControls) -> GenerationControls {
    GenerationControls {
        max_output_tokens: None,
        sampling,
        stops: Vec::new(),
        end_of_generation: EndOfGeneration::Stop,
        reasoning_budget: None,
        prompt_cache: PromptCache::Allowed,
    }
}

fn identity_sampling() -> SamplingControls {
    SamplingControls {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: 0,
    }
}

fn limits(tokenizer: &ByteBpeTokenizer, context_tokens: usize) -> ModelLimits {
    ModelLimits {
        context_tokens,
        vocabulary: tokenizer.vocabulary(),
        output_capacity: 8,
        forced_quantum: 4,
        method: MethodPolicy::Plain,
    }
}

#[test]
fn generation_options_forward_all_sampling_shaping_options() {
    let tokenizer = tokenizer();
    let prepared = prepared(&tokenizer, "");
    let options = generation_options(
        &prepared,
        prepared.prompt_tokens(),
        &controls(SamplingControls {
            temperature: 0.5,
            top_p: 0.8,
            top_k: 17,
            min_p: 0.1,
            repetition_penalty: 1.2,
            presence_penalty: -0.3,
            frequency_penalty: 0.4,
            seed: 9,
        }),
        &tokenizer,
        &limits(&tokenizer, 128),
    )
    .unwrap();
    assert_eq!(options.sampling, magnitude_chat::Sampling::Categorical);
    assert_eq!(options.shaping.temperature, 0.5);
    assert_eq!(options.shaping.top_p, 0.8);
    assert_eq!(options.shaping.top_k, 17);
    assert_eq!(options.shaping.min_p, 0.1);
    assert_eq!(options.shaping.repetition_penalty, 1.2);
    assert_eq!(options.shaping.presence_penalty, -0.3);
    assert_eq!(options.shaping.frequency_penalty, 0.4);
    assert_eq!(options.seed, 9);
    // Without a requested limit, output is bounded only by the context.
    assert_eq!(options.max_tokens, 128 - prepared.prompt_tokens() + 1);

    let greedy = generation_options(
        &prepared,
        prepared.prompt_tokens(),
        &controls(SamplingControls {
            temperature: 0.0,
            ..identity_sampling()
        }),
        &tokenizer,
        &limits(&tokenizer, 128),
    )
    .unwrap();
    assert_eq!(greedy.sampling, magnitude_chat::Sampling::Greedy);

    for invalid in [
        SamplingControls {
            top_p: 0.0,
            ..identity_sampling()
        },
        SamplingControls {
            min_p: 1.1,
            ..identity_sampling()
        },
        SamplingControls {
            repetition_penalty: 0.0,
            ..identity_sampling()
        },
        SamplingControls {
            top_k: 16_777_217,
            ..identity_sampling()
        },
    ] {
        assert!(matches!(
            generation_options(
                &prepared,
                prepared.prompt_tokens(),
                &controls(invalid),
                &tokenizer,
                &limits(&tokenizer, 128),
            ),
            Err(ChatError::InvalidRequest(_))
        ));
    }
}

#[test]
fn a_prompt_must_leave_one_context_position_for_generation() {
    let tokenizer = tokenizer();
    let prepared = prepared(&tokenizer, "");
    let prompt = prepared.prompt_tokens();
    assert!(generation_options(
        &prepared,
        prompt,
        &controls(identity_sampling()),
        &tokenizer,
        &limits(&tokenizer, prompt + 1),
    )
    .is_ok());
    assert_eq!(
        generation_options(
            &prepared,
            prompt,
            &controls(identity_sampling()),
            &tokenizer,
            &limits(&tokenizer, prompt),
        )
        .unwrap_err(),
        ChatError::ContextLengthExceeded {
            prompt_tokens: prompt,
            context_tokens: prompt,
        }
    );
}

#[test]
fn a_reasoning_budget_needs_template_reasoning_delimiters() {
    let tokenizer = tokenizer();
    let prepared = prepared(&tokenizer, "");
    let mut budgeted = controls(identity_sampling());
    budgeted.reasoning_budget = std::num::NonZeroU32::new(8);
    assert!(matches!(
        generation_options(
            &prepared,
            prepared.prompt_tokens(),
            &budgeted,
            &tokenizer,
            &limits(&tokenizer, 128),
        ),
        Err(ChatError::InvalidRequest(_))
    ));
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

/// LFM, Gemma and Muse templates render the BOS text while their
/// tokenizers also begin every sequence with the BOS; the prompt holds it once.
#[test]
fn implicit_bos_begins_every_prompt_exactly_once() {
    let bos_first = ByteBpeTokenizer::new(BpeConfig {
        implicit_bos: Some(TokenId(257)),
        ..config()
    })
    .unwrap();
    let request = ChatRequest::new(vec![serde_json::json!({"role":"user", "content":"é"})], 0);
    let prepare = |source: &str| {
        let bundle = TemplateBundle::new(
            vec![TemplateVariant {
                name: "default".into(),
                source: source.into(),
                provenance: "fixture".into(),
            }],
            "default".into(),
            [("bos_token".to_string(), "<bos>".to_string())].into(),
        )
        .unwrap();
        PreparedChat::prepare(&bundle, &bos_first, &request, &TemplateSelection::default())
            .unwrap()
            .input()
            .tokens
            .clone()
    };
    let expected = vec![TokenId(257), TokenId(195), TokenId(169)];
    assert_eq!(
        prepare("{{ bos_token }}{{ messages[0].content }}"),
        expected
    );
    assert_eq!(prepare("<bos>{{ messages[0].content }}"), expected);
    assert_eq!(prepare("{{ messages[0].content }}"), expected);
    // Without an implicit BOS, the rendered text alone decides.
    let plain = PreparedChat::prepare(
        &bundle(""),
        &tokenizer(),
        &request,
        &TemplateSelection::default(),
    )
    .unwrap();
    assert_eq!(plain.input().tokens, vec![TokenId(195), TokenId(169)]);
    // Fragments never receive the sequence start.
    assert_eq!(
        bos_first.encode("é", SpecialTokens::Recognize).unwrap(),
        vec![TokenId(195), TokenId(169)]
    );
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
    use magnitude_chat::{Options, Sampling};
    use magnitude_generation::{GenerationSeed, Plain, RequestId, RoundStart, Shaping, WorkKind};
    use magnitude_grammar::{CacheLimits, Vocabulary};
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
    let layout = InputLayout::new(tokens.len(), Vec::new()).unwrap();
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
            suppressed_tokens: tokenizer.suppressed_tokens().clone(),
            sampling: Sampling::Greedy,
            shaping: Shaping {
                temperature: 0.0,
                ..Default::default()
            },
            seed: 0,
            forced_quantum,
            method: magnitude_generation::MethodChoice::Plain,
            end_of_generation: EndOfGeneration::Stop,
            reasoning_budget: None,
        };
        let plan = prepared.input().constraint.as_ref().unwrap();
        let constraint = vocabulary.bind(&plan.grammar, &plan.prefix).unwrap();
        let mut generation = GenerationSeed::new(
            tokens.clone(),
            layout.clone(),
            options,
            Some(Box::new(constraint)),
        )
        .unwrap()
        .into_generation(Arc::new(Plain))
        .unwrap();
        // The executor installs fresh numerical state before the first round.
        generation.resume_at(None).unwrap();
        let Ok(RoundStart::Target(round)) = generation.start_round(RequestId(1), 64) else {
            panic!("a fresh plain generation starts a target round");
        };
        let forward = round.round_forward();
        assert_eq!(forward.kind, WorkKind::Prefill);
        assert_eq!(forward.tokens, tokens);
        // The schema admits leading whitespace, so no token is forced: the
        // first output is selected under the grammar mask.
        assert_eq!(forward.selects.len(), 1);
        assert!(forward.selects[0].mask.is_some());
    }
}

fn conversation(system: &str, question: &str) -> ChatRequest {
    ChatRequest::new(
        vec![
            serde_json::json!({"role": "system", "content": system}),
            serde_json::json!({"role": "user", "content": question}),
        ],
        0,
    )
}

fn tagged(source: &str) -> TemplateBundle {
    TemplateBundle::new(
        vec![TemplateVariant {
            name: "default".into(),
            source: source.into(),
            provenance: "fixture".into(),
        }],
        "default".into(),
        Default::default(),
    )
    .unwrap()
}

/// Each turn opens with the `<bos>` control token and closes with `<eos>`.
const TURNS: &str =
    "{% for m in messages %}<bos>{{ m.role }}: {{ m.content }}<eos>{% endfor %}<bos>assistant: ";

/// The last message's boundary is the position after the control token
/// that opens it. Every request differing only in the last message's
/// content shares the tokens before it, whatever the content starts with,
/// including content that begins like the probe.
#[test]
fn last_message_boundary_follows_the_token_that_opens_it() {
    let tokenizer = tokenizer();
    let bundle = tagged(TURNS);
    let system = "You answer tersely and cite the harbour log. ".repeat(8);
    let selection = TemplateSelection::default();
    let boundary = tokenizer
        .encode(&format!("<bos>system: {system}<eos><bos>"), SpecialTokens::Recognize)
        .unwrap()
        .len();
    let questions = ["Hello there", "Hello again", "Why is the sky blue?", "\u{E000}x"];
    let prepared = questions.map(|question| {
        let request = conversation(&system, question);
        let chat = PreparedChat::prepare(&bundle, &tokenizer, &request, &selection).unwrap();
        let found = chat.last_message_boundary(&bundle, &request, &selection);
        (chat, found)
    });
    for (chat, found) in &prepared {
        assert_eq!(*found, Some(boundary));
        assert_eq!(
            chat.input().tokens[..boundary],
            prepared[0].0.input().tokens[..boundary]
        );
    }
}

/// A template that renders an assistant turn in history unlike its
/// generation prompt (here an empty reasoning block the history omits):
/// the next turn shares the prompt through the token opening the assistant
/// turn, short of the prompt's end, whatever the reply and next message are.
#[test]
fn next_turn_boundary_follows_the_last_token_the_next_turn_shares() {
    let tokenizer = tokenizer();
    let bundle = tagged(
        "{% for m in messages %}<bos>{{ m.role }}: {{ m.content }}<eos>{% endfor %}<bos>assistant: <eos>think<eos>",
    );
    let selection = TemplateSelection::default();
    let request = conversation("be terse", "Hello there");
    let chat = PreparedChat::prepare(&bundle, &tokenizer, &request, &selection).unwrap();
    let boundary = tokenizer
        .encode("<bos>system: be terse<eos><bos>user: Hello there<eos><bos>", SpecialTokens::Recognize)
        .unwrap()
        .len();
    assert!(boundary < chat.input().tokens.len());
    assert_eq!(chat.next_turn_boundary(&bundle, &request, &selection), Some(boundary));
    for (reply, question) in [("Hi", "And now?"), ("\u{E000}", "x")] {
        let mut next = request.clone();
        next.messages.push(serde_json::json!({"role": "assistant", "content": reply}));
        next.messages.push(serde_json::json!({"role": "user", "content": question}));
        let next = PreparedChat::prepare(&bundle, &tokenizer, &next, &selection).unwrap();
        assert_eq!(next.input().tokens[..boundary], chat.input().tokens[..boundary]);
        assert_ne!(next.input().tokens[boundary + 11], chat.input().tokens[boundary + 11]);
    }
    // A template whose history continues its generation prompt shares it
    // whole: the boundary follows the prompt's last added token.
    let plain = tagged(TURNS);
    let chat = PreparedChat::prepare(&plain, &tokenizer, &request, &selection).unwrap();
    assert_eq!(chat.next_turn_boundary(&plain, &request, &selection), Some(boundary));
}

/// An inserted sequence-start token shifts the boundary by one position.
#[test]
fn last_message_boundary_counts_the_inserted_sequence_start() {
    let tokenizer = ByteBpeTokenizer::new(BpeConfig {
        implicit_bos: Some(TokenId(257)),
        ..config()
    })
    .unwrap();
    let bundle = tagged(&format!("chat\n{TURNS}"));
    let request = conversation("be terse", "Hello");
    let selection = TemplateSelection::default();
    let chat = PreparedChat::prepare(&bundle, &tokenizer, &request, &selection).unwrap();
    assert_eq!(chat.input().tokens[0], TokenId(257));
    let opened = tokenizer
        .encode("chat\n<bos>system: be terse<eos><bos>", SpecialTokens::Recognize)
        .unwrap()
        .len();
    assert_eq!(
        chat.last_message_boundary(&bundle, &request, &selection),
        Some(1 + opened)
    );
}

/// Without a rendered content, or without a token opening the message,
/// there is no boundary.
#[test]
fn last_message_boundary_needs_rendered_content_after_an_added_token() {
    let tokenizer = tokenizer();
    let selection = TemplateSelection::default();
    let request = conversation("system", "question");
    for source in [
        "{% for m in messages %}<bos>{{ m.role }}<eos>{% endfor %}",
        "{% for m in messages %}[{{ m.role }}] {{ m.content }}\n{% endfor %}",
    ] {
        let bundle = tagged(source);
        let chat = PreparedChat::prepare(&bundle, &tokenizer, &request, &selection).unwrap();
        assert_eq!(chat.last_message_boundary(&bundle, &request, &selection), None);
    }
}
