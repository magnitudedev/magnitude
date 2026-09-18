use seismic_engine::inputs::{
    BoundaryRule, BpeConfig, ByteBpeTokenizer, InputLayout, InputSpan, PieceKind, SpecialTokens,
    TokenId,
};
use std::collections::BTreeSet;

fn byte_config() -> BpeConfig {
    let mut bytes: Vec<u8> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut alphabet: Vec<u32> = bytes.iter().map(|&b| u32::from(b)).collect();
    let mut next = 256;
    for b in 0..=255 {
        if !bytes.contains(&b) {
            bytes.push(b);
            alphabet.push(next);
            next += 1;
        }
    }
    let mut pieces = vec![String::new(); 256];
    for (b, c) in bytes.into_iter().zip(alphabet) {
        pieces[b as usize] = char::from_u32(c).unwrap().to_string();
    }
    let mut kinds = vec![PieceKind::Normal; 256];
    pieces.push("<eos>".into());
    kinds.push(PieceKind::Control);
    pieces.push("<user>".into());
    kinds.push(PieceKind::UserDefined);
    BpeConfig {
        artifact_identity: "fixture".into(),
        pieces,
        kinds,
        merges: vec![],
        pattern: r".+|\s".into(),
        normalize_nfc: true,
        stop_tokens: BTreeSet::from([TokenId(256)]),
    }
}

#[test]
fn byte_bpe_preserves_ids_special_policy_and_incremental_utf8() {
    let tokenizer = ByteBpeTokenizer::new(byte_config()).unwrap();
    let text = "café 世界 🦙";
    let encoded = tokenizer.encode(text, SpecialTokens::Recognize).unwrap();
    assert_eq!(
        encoded,
        text.as_bytes()
            .iter()
            .map(|&b| TokenId(u32::from(b)))
            .collect::<Vec<_>>()
    );
    let mut decoder = tokenizer.decoder(true);
    let mut decoded = String::new();
    for token in encoded {
        decoded.push_str(&decoder.push(token).unwrap());
    }
    decoded.push_str(&decoder.finish().unwrap());
    assert_eq!(decoded, text);
    assert!(decoder.push(TokenId(1)).is_err());
    assert_eq!(
        tokenizer
            .encode("<eos><user>", SpecialTokens::Recognize)
            .unwrap(),
        vec![TokenId(256), TokenId(257)]
    );
    let literal = tokenizer
        .encode("<eos><user>", SpecialTokens::Literal)
        .unwrap();
    assert!(!literal.contains(&TokenId(256)));
    assert_eq!(literal.last(), Some(&TokenId(257)));
    assert_eq!(tokenizer.piece(TokenId(256), true).unwrap(), b"");
    assert_eq!(tokenizer.piece(TokenId(256), false).unwrap(), b"<eos>");
    assert_eq!(
        tokenizer
            .encode("e\u{301}", SpecialTokens::Literal)
            .unwrap(),
        vec![TokenId(195), TokenId(169)]
    );
}

#[test]
fn replacement_decode_retains_only_incomplete_suffix_and_rejects_bad_ids() {
    let tokenizer = ByteBpeTokenizer::new(byte_config()).unwrap();
    let mut decoder = tokenizer.decoder(true);
    assert_eq!(decoder.push(TokenId(0xe4)).unwrap(), "");
    assert!(decoder.push(TokenId(999)).is_err());
    assert_eq!(decoder.push(TokenId(b'x' as u32)).unwrap(), "�x");
    assert_eq!(decoder.push(TokenId(0xff)).unwrap(), "�");
    assert_eq!(decoder.push(TokenId(0xf0)).unwrap(), "");
    assert_eq!(decoder.push(TokenId(0x9f)).unwrap(), "");
    assert_eq!(decoder.finish().unwrap(), "�");
    let mut config = byte_config();
    config.pieces[1] = config.pieces[0].clone();
    assert!(ByteBpeTokenizer::new(config).is_err());
    let mut config = byte_config();
    config.kinds[1] = PieceKind::Unused;
    assert!(ByteBpeTokenizer::new(config).is_err());
}

#[test]
fn input_spans_bound_advances_without_using_physical_page_geometry() {
    let layout = InputLayout::new(
        10,
        vec![InputSpan {
            start: 2,
            end: 8,
            identity: "image-a".into(),
            boundaries: BoundaryRule::Indivisible,
            language_history: false,
        }],
    )
    .unwrap();
    assert_eq!(layout.chunk_end(0, 10, 4).unwrap(), 2);
    assert_eq!(layout.chunk_end(2, 10, 1).unwrap(), 8);
    assert_eq!(layout.chunk_end(8, 10, 4).unwrap(), 10);
    assert!(layout.chunk_end(3, 10, 1).is_err());
    assert!(layout.chunk_end(0, 3, 1).is_err());
    assert!(!layout.language(3).unwrap());
    assert!(layout.language(10).unwrap());
    assert!(layout.boundary(11));
    assert!(InputLayout::new(4, layout.spans().to_vec()).is_err());
}

#[test]
fn artifact_metadata_uses_file_precedence_and_preserves_token_ids() {
    use seismic_engine::{chat::TemplateSelection, inputs::artifacts};
    use serde_json::json;
    let path = std::env::temp_dir().join(format!("seismic-input-artifacts-{}", std::process::id()));
    std::fs::create_dir_all(path.join("chat_templates")).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(path.clone());
    std::fs::write(
        path.join("processor_config.json"),
        json!({"chat_template":{"default":"processor","tool_use":"processor tools"}}).to_string(),
    )
    .unwrap();
    std::fs::write(
        path.join("tokenizer_config.json"),
        json!({"chat_template":"tokenizer","eos_token":{"content":"<eos>"}}).to_string(),
    )
    .unwrap();
    std::fs::write(path.join("chat_templates/tool_use.jinja"), "named tools").unwrap();
    std::fs::write(path.join("chat_template.jinja"), "final default").unwrap();
    let templates = artifacts::directory_templates(&path).unwrap();
    assert_eq!(
        templates
            .select(false, &TemplateSelection::default())
            .unwrap()
            .source,
        "final default"
    );
    assert_eq!(
        templates
            .select(true, &TemplateSelection::default())
            .unwrap()
            .source,
        "named tools"
    );
    let config = byte_config();
    let mut data = json!({
        "model":{"type":"BPE","byte_fallback":false,"unk_token":null,"vocab":config.pieces.iter().enumerate().map(|(i,p)|(p.clone(),json!(i))).collect::<serde_json::Map<_,_>>(),"merges":[]},
        "normalizer":{"type":"NFC"},
        "pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":config.pattern},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},
        "added_tokens":[{"id":256,"content":"<eos>","special":true,"normalized":false,"single_word":false,"lstrip":false,"rstrip":false},{"id":257,"content":"<user>","special":false,"normalized":false,"single_word":false,"lstrip":false,"rstrip":false}]
    });
    std::fs::write(path.join("tokenizer.json"), data.to_string()).unwrap();
    let tokenizer =
        ByteBpeTokenizer::new(artifacts::mlx_tokenizer(&path, "fixture".into()).unwrap()).unwrap();
    assert_eq!(
        tokenizer
            .encode("<eos>é", SpecialTokens::Recognize)
            .unwrap(),
        vec![TokenId(256), TokenId(195), TokenId(169)]
    );
    data["added_tokens"][0]["content"] = json!("changed identity");
    std::fs::write(path.join("tokenizer.json"), data.to_string()).unwrap();
    assert!(artifacts::mlx_tokenizer(&path, "fixture".into()).is_err());
    data["added_tokens"][0]["content"] = json!("<eos>");
    data["model"]["dropout"] = json!(0.1);
    std::fs::write(path.join("tokenizer.json"), data.to_string()).unwrap();
    assert!(artifacts::mlx_tokenizer(&path, "fixture".into()).is_err());
}

#[test]
fn prepared_chat_uses_the_same_prompt_for_counting_and_model_input() {
    use seismic_engine::chat::{
        ChatRequest, PreparedChat, TemplateBundle, TemplateSelection, TemplateVariant,
    };
    let tokenizer = ByteBpeTokenizer::new(byte_config()).unwrap();
    let bundle = TemplateBundle::new(
        vec![TemplateVariant {
            name: "default".into(),
            source: "{{ messages[0].content }}<eos>".into(),
            provenance: "fixture".into(),
        }],
        "default".into(),
        Default::default(),
    )
    .unwrap();
    let request = ChatRequest::new(vec![serde_json::json!({"role":"user","content":"é"})], 0);
    let prepared =
        PreparedChat::prepare(&bundle, &tokenizer, &request, &TemplateSelection::default())
            .unwrap();
    assert_eq!(prepared.prompt(), "é<eos>");
    assert_eq!(prepared.prompt_tokens(), 3);
    assert_eq!(
        prepared.input().tokens,
        vec![TokenId(195), TokenId(169), TokenId(256)]
    );
    assert_eq!(prepared.input().tokenizer_identity, tokenizer.identity());
    assert!(prepared.input().constraint.is_none());
}

#[test]
fn accepted_tokens_decode_stop_and_finish_through_one_native_stream() {
    use seismic_engine::{
        chat::{
            ChatRequest, Event, PreparedChat, TemplateBundle, TemplateSelection, TemplateVariant,
            TerminalCause, TokenChatStream,
        },
        generation::{FinishReason, OutputToken},
    };
    let tokenizer = ByteBpeTokenizer::new(byte_config()).unwrap();
    let bundle = TemplateBundle::new(
        vec![TemplateVariant {
            name: "default".into(),
            source: "{{ messages[0].content }}".into(),
            provenance: "fixture".into(),
        }],
        "default".into(),
        Default::default(),
    )
    .unwrap();
    let request = ChatRequest::new(
        vec![serde_json::json!({"role":"user","content":"hello"})],
        0,
    );
    let prepared =
        PreparedChat::prepare(&bundle, &tokenizer, &request, &TemplateSelection::default())
            .unwrap();
    let mut stream =
        TokenChatStream::new(&prepared, &tokenizer, vec!["世界".into()], 4096).unwrap();
    let mut events = Vec::new();
    for (index, &byte) in "café 世界 ignored".as_bytes().iter().enumerate() {
        events.extend(
            stream
                .feed(&OutputToken {
                    index,
                    token: TokenId(byte as u32),
                })
                .unwrap(),
        );
        if stream.stopped() {
            break;
        }
    }
    let content: String = events
        .iter()
        .filter_map(|e| {
            if let Event::Content { text } = e {
                Some(text.as_str())
            } else {
                None
            }
        })
        .collect();
    assert_eq!(content, "café ");
    assert_eq!(
        events.last(),
        Some(&Event::Finish {
            cause: TerminalCause::UserStop
        })
    );
    assert!(stream.finish(FinishReason::Cancelled).is_err());

    let mut stream = TokenChatStream::new(&prepared, &tokenizer, vec![], 4096).unwrap();
    assert!(stream
        .feed(&OutputToken {
            index: 0,
            token: TokenId(0xe2)
        })
        .unwrap()
        .is_empty());
    let events = stream.finish(FinishReason::Length).unwrap();
    assert!(events.contains(&Event::Content { text: "�".into() }));
    assert_eq!(
        events.last(),
        Some(&Event::Finish {
            cause: TerminalCause::Length
        })
    );

    let mut stream = TokenChatStream::new(&prepared, &tokenizer, vec![], 4096).unwrap();
    assert!(stream
        .feed(&OutputToken {
            index: 1,
            token: TokenId(65)
        })
        .is_err());
    assert!(stream
        .feed(&OutputToken {
            index: 0,
            token: TokenId(65)
        })
        .is_err());
    let mut other = byte_config();
    other.artifact_identity = "different".into();
    let other = ByteBpeTokenizer::new(other).unwrap();
    assert!(TokenChatStream::new(&prepared, &other, vec![], 4096).is_err());
}
