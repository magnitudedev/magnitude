use magnitude_artifacts::InputLayout;
use magnitude_chat::{Normalization, PieceEncoding, Split, SplitBehavior};
use magnitude_engine::{
    generation::Constraint,
    inputs::{BpeConfig, ByteBpeTokenizer, PieceKind, SpecialTokens, TokenId},
    worker::GenerationBinding,
};
use magnitude_family_contracts::{PreparedModelInput, TokenPlan};
use magnitude_grammar::{compile, CacheLimits, Grammar, Vocabulary};
use serde_json::json;
use std::{collections::BTreeSet, sync::Arc};

fn text_input(tokens: Vec<TokenId>) -> PreparedModelInput {
    let coordinates = (0..tokens.len())
        .map(|position| [position as i32; 3])
        .collect();
    let layout = InputLayout::new(tokens.len(), Vec::new()).unwrap();
    PreparedModelInput::from_text_coordinates(TokenPlan::new(tokens, layout).unwrap(), coordinates)
        .unwrap()
}

fn tokenizer() -> Arc<ByteBpeTokenizer> {
    // Independent byte alphabet fixture: IDs 0..255 are exactly their byte value.
    let included: BTreeSet<u8> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let missing = (0..=255)
        .filter(|b| !included.contains(b))
        .collect::<Vec<_>>();
    let mut pieces = (0..=255)
        .map(|b| {
            char::from_u32(if included.contains(&b) {
                u32::from(b)
            } else {
                256 + missing.iter().position(|&v| v == b).unwrap() as u32
            })
            .unwrap()
            .to_string()
        })
        .collect::<Vec<_>>();
    pieces.extend(["<eos>".into(), "<end>".into(), "<unused>".into()]);
    let mut kinds = vec![PieceKind::Normal; 256];
    kinds.extend([PieceKind::Control, PieceKind::Control, PieceKind::Unused]);
    Arc::new(
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
            stop_tokens: BTreeSet::from([TokenId(256), TokenId(257)]),
            suppressed_tokens: BTreeSet::new(),
        })
        .unwrap(),
    )
}

fn grammar(gbnf: &str) -> Grammar {
    compile(gbnf).unwrap().grammar
}

fn vocabulary(tokenizer: Arc<ByteBpeTokenizer>) -> Vocabulary {
    Vocabulary::new(
        tokenizer,
        272,
        CacheLimits {
            entries: 2,
            bytes: 1024 * 1024,
        },
    )
    .unwrap()
}

fn allowed(mask: &[u32], token: u32) -> bool {
    mask[token as usize / 32] & (1 << (token % 32)) != 0
}

#[test]
fn exact_masks_forks_staging_eos_and_projection_padding() {
    let tokenizer = tokenizer();
    let mut vocab = vocabulary(tokenizer.clone());
    let initial = vocab
        .bind(&grammar("root ::= \"a\" (\"b\" | \"c\")"), &[])
        .unwrap();
    let mask = initial.mask().unwrap();
    assert!(allowed(&mask, 97));
    assert!(!allowed(&mask, 98));
    for id in [255, 256, 257, 258, 259, 271] {
        assert!(!allowed(&mask, id));
    }
    assert!(initial.advance(&[TokenId(98)]).is_err());
    let a = initial.advance(&[TokenId(97)]).unwrap();
    assert_eq!(initial.position(), 0);
    assert_eq!(a.position(), 1);
    assert!(allowed(&initial.mask().unwrap(), 97));
    assert!(allowed(&a.mask().unwrap(), 98));
    assert!(allowed(&a.mask().unwrap(), 99));
    assert!(a.advance(&[TokenId(256)]).is_err());
    let end = a.advance(&[TokenId(98)]).unwrap();
    assert!(end.accepting().unwrap());
    for eos in [256, 257] {
        assert!(allowed(&end.mask().unwrap(), eos));
        let terminal = end.advance(&[TokenId(eos)]).unwrap();
        assert!(terminal.stopped());
        assert!(terminal.advance(&[TokenId(97)]).is_err());
    }
    assert!(initial.advance(&[TokenId(256), TokenId(97)]).is_err());
    assert!(initial.advance(&[TokenId(258)]).is_err());
}

#[test]
fn binding_uses_only_declared_prefix_and_bounds_cache_by_grammar() {
    let tokenizer = tokenizer();
    let mut vocab = vocabulary(tokenizer.clone());
    let compiled = grammar("root ::= \"prefix:ab\"");
    let prefix = tokenizer
        .encode("prefix:", SpecialTokens::Recognize)
        .unwrap();
    let initial = vocab.bind(&compiled, &prefix).unwrap();
    assert_eq!(initial.position(), 0);
    assert_eq!(initial.forced(1).unwrap(), vec![TokenId(97)]);
    let advanced = initial.advance(&[TokenId(97)]).unwrap();
    assert_eq!(advanced.forced(5).unwrap(), vec![TokenId(98)]);
    let cached = vocab.bind(&compiled, &prefix).unwrap();
    assert_eq!(cached.position(), 0);
    assert_eq!(vocab.cache_hits(), 1);
    assert!(allowed(&cached.mask().unwrap(), 97));
    let conversation = tokenizer
        .encode("entire conversation prefix:", SpecialTokens::Recognize)
        .unwrap();
    assert!(vocab.bind(&compiled, &conversation).is_err());
    for text in ["x", "y", "z"] {
        vocab
            .bind(&grammar(&format!("root ::= {text:?}")), &[])
            .unwrap();
    }
    assert_eq!(vocab.cached_entries(), 2);
    assert!(vocab.cached_bytes() <= 1024 * 1024);
}

#[test]
fn native_qwen_grammars_enforce_required_tool_names_and_argument_schemas() {
    use magnitude_templates::{Request, Template, ToolChoice};
    let tokenizer = tokenizer();
    let mut vocab = vocabulary(tokenizer.clone());
    for (source,valid,invalid) in [
        (include_str!("../templates/tests/assets/Qwen-Qwen3-0.6B.jinja"),
         "<tool_call>\n{\"name\":\"search\",\"arguments\":{\"query\":\"héllo 世界\"}}\n</tool_call>",
         "<tool_call>\n{\"name\":\"other\",\"arguments\":{\"query\":\"x\"}}\n</tool_call>"),
        (include_str!("../templates/tests/assets/Qwen3.5-4B.jinja"),
         "<tool_call>\n<function=search>\n<parameter=query>\nhéllo 世界\n</parameter>\n</function>\n</tool_call>",
         "<tool_call>\n<function=other>\n<parameter=query>\nx\n</parameter>\n</function>\n</tool_call>"),
    ] {
        let template=Template::new(source,&Default::default()).unwrap();
        let mut request=Request::new(vec![json!({"role":"user","content":"search"})],0);
        request.tool_choice=ToolChoice::Required;
        request.template_arguments.insert("enable_thinking".into(),json!(false));
        request.tools=vec![json!({"type":"function","function":{"name":"search","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"],"additionalProperties":false}}})];
        let prepared=template.prepare(&request).unwrap();let d=prepared.description();
        let compiled = compile(&d.grammar).unwrap();
        assert_eq!(compiled.report.character_lexemes, 0, "{:?}", compiled.report);
        let prefix = tokenizer.encode(&d.grammar_initial_prefix, SpecialTokens::Recognize).unwrap();
        let initial=vocab.bind(&compiled.grammar,&prefix).unwrap();
        let tokens=tokenizer.encode(valid,SpecialTokens::Recognize).unwrap();
        let completed=initial.advance(&tokens).unwrap();assert!(completed.accepting().unwrap());
        assert!(initial.advance(&tokenizer.encode(invalid,SpecialTokens::Recognize).unwrap()).is_err());
        // Some native templates permit prose before the required call. Such
        // a prefix must never authorize completion without that call.
        if let Ok(prose) = initial.advance(&tokenizer.encode("ordinary prose",SpecialTokens::Recognize).unwrap()) {
            assert!(!prose.accepting().unwrap());
            assert!(prose.advance(&[TokenId(256)]).is_err());
        }
    }
}

#[test]
fn host_preparation_binds_before_generation_and_rejects_identity_mismatches() {
    use magnitude_chat::{
        ChatRequest, EndOfGeneration, PreparedChat, ReasoningIntent, TemplateBundle,
        TemplateSelection, TemplateVariant,
    };
    use magnitude_engine::generation::{MethodChoice, Options, Sampling, Shaping};
    let tokenizer = tokenizer();
    let bundle = TemplateBundle::new(
        vec![TemplateVariant {
            name: "default".into(),
            source: include_str!("../templates/tests/assets/Qwen-Qwen3-0.6B.jinja").into(),
            provenance: "fixture".into(),
        }],
        "default".into(),
        Default::default(),
    )
    .unwrap();
    let mut request = ChatRequest::new(vec![json!({"role":"user","content":"say hello"})], 0);
    request.json_schema = Some(
        json!({"type":"object","properties":{"message":{"type":"string"}},"required":["message"],"additionalProperties":false}),
    );
    request.reasoning = ReasoningIntent::Disabled;
    let prepared =
        PreparedChat::prepare(&bundle, &tokenizer, &request, &TemplateSelection::default())
            .unwrap();
    let limits = CacheLimits {
        entries: 2,
        bytes: 1024 * 1024,
    };
    let mut binding = GenerationBinding::new(tokenizer.clone(), 272, limits).unwrap();
    let options = Options {
        max_tokens: 128,
        output_capacity: 16,
        context_limit: 4096,
        vocabulary: 272,
        stop_tokens: tokenizer.stop_tokens().clone(),
        suppressed_tokens: tokenizer.suppressed_tokens().clone(),
        sampling: Sampling::Greedy,
        shaping: Shaping {
            temperature: 0.0,
            ..Default::default()
        },
        seed: 0,
        forced_quantum: 16,
        method: MethodChoice::Plain,
        end_of_generation: EndOfGeneration::Stop,
        reasoning_budget: None,
    };
    let plan = prepared.input().constraint.as_ref();
    let text = text_input(prepared.input().tokens.clone());
    let generation = binding
        .seed(plan, options.clone(), text.tokens(), text.layout())
        .unwrap();
    assert_eq!(generation.constraint_position(), Some(0));
    assert!(generation.selection_mask().unwrap().is_some());
    let mut interpreted_tokens = prepared.input().tokens.clone();
    interpreted_tokens.push(prepared.input().tokens[0]);
    let interpreted = text_input(interpreted_tokens.clone());
    let interpreted_generation = binding
        .seed(
            plan,
            options.clone(),
            interpreted.tokens(),
            interpreted.layout(),
        )
        .unwrap();
    assert_eq!(interpreted_generation.prompt(), interpreted_tokens);
    let mut bad = plan.unwrap().clone();
    bad.tokenizer_identity = "another tokenizer".into();
    assert!(binding
        .seed(Some(&bad), options.clone(), text.tokens(), text.layout())
        .is_err());
    let mut bad_options = options.clone();
    bad_options.stop_tokens.clear();
    assert!(binding
        .seed(plan, bad_options, text.tokens(), text.layout())
        .is_err());

    let mut mtp_options = options;
    mtp_options.method = MethodChoice::Mtp { proposals: 2 };
    let mut mtp_binding = GenerationBinding::new(tokenizer, 272, limits).unwrap();
    let mtp = mtp_binding
        .seed(plan, mtp_options, text.tokens(), text.layout())
        .unwrap();
    assert_eq!(mtp.method(), MethodChoice::Mtp { proposals: 2 });
}

#[test]
fn invalid_caller_grammars_are_rejected_at_preparation() {
    use magnitude_chat::{
        ChatError, ChatRequest, PreparedChat, TemplateBundle, TemplateSelection, TemplateVariant,
    };
    let tokenizer = tokenizer();
    let bundle = TemplateBundle::new(
        vec![TemplateVariant {
            name: "default".into(),
            source: include_str!("../templates/tests/assets/Qwen-Qwen3-0.6B.jinja").into(),
            provenance: "fixture".into(),
        }],
        "default".into(),
        Default::default(),
    )
    .unwrap();
    let mut request = ChatRequest::new(vec![json!({"role":"user","content":"say hello"})], 0);
    request.grammar = Some("root ::= missing".into());
    assert!(matches!(
        PreparedChat::prepare(&bundle, &tokenizer, &request, &TemplateSelection::default()),
        Err(ChatError::InvalidRequest(_))
    ));
}
