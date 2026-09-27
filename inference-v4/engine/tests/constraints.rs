use magnitude_artifacts::InputLayout;
use magnitude_engine::{
    chat::ConstraintPlan,
    chat::{CacheLimits, Vocabulary},
    generation::{
        grammar::{to_lark, CONVERTER_IDENTITY},
        Constraint,
    },
    inputs::{BpeConfig, ByteBpeTokenizer, PieceKind, SpecialTokens, TokenId},
};
use magnitude_model_contracts::{PreparedModelInput, TokenPlan};
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
            pattern: r".+|\s".into(),
            normalize_nfc: false,
            stop_tokens: BTreeSet::from([TokenId(256), TokenId(257)]),
        })
        .unwrap(),
    )
}
fn plan(tokenizer: &ByteBpeTokenizer, gbnf: &str, prefix: &str) -> ConstraintPlan {
    ConstraintPlan {
        artifact_identity: tokenizer.artifact_identity().into(),
        tokenizer_identity: tokenizer.identity().into(),
        template_identity: "fixture-template".into(),
        converter_identity: CONVERTER_IDENTITY.into(),
        gbnf: gbnf.into(),
        initial_prefix: prefix.into(),
    }
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
fn parsed_rules_preserve_unicode_classes_repetitions_and_name_collisions() {
    let tokenizer = tokenizer();
    let mut vocab = vocabulary(tokenizer.clone());
    let grammars=[
        ("root ::= start A a\nstart ::= \"\\u000a\"\nA ::= [\\x61-\\u0062]{1,2}\na ::= \"\\U0001f999\"\n",vec!["\na🦙","\nbb🦙"],vec!["na🦙","\nccc🦙","\na"]),
        ("root ::= (\"a\" |\n \"b\")+ [^x-z]? # comment\n",vec!["a","abba!","b\n"],vec!["","abz","xx"]),
        ("root ::= .{2,3}\n",vec!["é🦙","\n\n","abc"],vec!["a","abcd"]),
        ("root ::= \"a\"{2,}\n",vec!["aa","aaaa"],vec!["","a","aab"]),
    ];
    for (grammar, accepted, rejected) in grammars {
        let initial = vocab.bind(&plan(&tokenizer, grammar, "")).unwrap();
        let accepts = |text: &str| {
            let tokens = tokenizer.encode(text, SpecialTokens::Recognize).unwrap();
            let result = if tokens.is_empty() {
                Ok(initial.fork())
            } else {
                initial.advance(&tokens)
            };
            result.and_then(|s| s.accepting()).unwrap_or(false)
        };
        for text in accepted {
            assert!(accepts(text), "{grammar}: {text:?}");
        }
        for text in rejected {
            assert!(!accepts(text), "{grammar}: {text:?}");
        }
    }
    for invalid in [
        "root ::= missing",
        "root ::= \"\\uD800\"",
        "root ::= [z-a]",
        "root ::= \"x\"{3,2}",
        "root ::= (\"x\"",
        "root ::= \"x\"\nroot ::= \"y\"",
        "other ::= \"x\"",
    ] {
        assert!(to_lark(invalid).is_err(), "{invalid}");
    }
}

#[test]
fn exact_masks_forks_staging_eos_and_projection_padding() {
    let tokenizer = tokenizer();
    let mut vocab = vocabulary(tokenizer.clone());
    let initial = vocab
        .bind(&plan(&tokenizer, "root ::= \"a\" (\"b\" | \"c\")", ""))
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
fn binding_uses_only_declared_prefix_and_bounds_cache_by_identity() {
    let tokenizer = tokenizer();
    let mut vocab = vocabulary(tokenizer.clone());
    let p = plan(&tokenizer, "root ::= \"prefix:ab\"", "prefix:");
    let initial = vocab.bind(&p).unwrap();
    assert_eq!(initial.position(), 0);
    assert_eq!(initial.forced(1).unwrap(), vec![TokenId(97)]);
    let advanced = initial.advance(&[TokenId(97)]).unwrap();
    assert_eq!(advanced.forced(5).unwrap(), vec![TokenId(98)]);
    let cached = vocab.bind(&p).unwrap();
    assert_eq!(cached.position(), 0);
    assert_eq!(vocab.cache_hits(), 1);
    assert!(allowed(&cached.mask().unwrap(), 97));
    let mut incompatible = p.clone();
    incompatible.tokenizer_identity = "other".into();
    assert!(vocab.bind(&incompatible).is_err());
    incompatible = p.clone();
    incompatible.converter_identity = "other".into();
    assert!(vocab.bind(&incompatible).is_err());
    incompatible = p.clone();
    incompatible.initial_prefix = "entire conversation prefix:".into();
    assert!(vocab.bind(&incompatible).is_err());
    for text in ["x", "y", "z"] {
        vocab
            .bind(&plan(&tokenizer, &format!("root ::= {text:?}"), ""))
            .unwrap();
    }
    assert_eq!(vocab.cached_entries(), 2);
    assert!(vocab.cached_bytes() <= 1024 * 1024);
}

#[test]
fn recursive_scanners_preserve_cfg_ambiguity_and_nested_structure() {
    let tokenizer = tokenizer();
    let mut vocab = vocabulary(tokenizer.clone());
    let grammar="root ::= scanner object\nscanner ::= [^<] scanner | \"<\"\nobject ::= \"(\" object \" )\" | \"value\"\n";
    let initial = vocab.bind(&plan(&tokenizer, grammar, "")).unwrap();
    let valid = "arbitrary scanner prefix <((value ) )";
    let tokens = tokenizer.encode(valid, SpecialTokens::Recognize).unwrap();
    assert!(initial.advance(&tokens).unwrap().accepting().unwrap());
    assert!(
        initial
            .advance(
                &tokenizer
                    .encode("prefix <(value", SpecialTokens::Recognize)
                    .unwrap()
            )
            .unwrap()
            .accepting()
            .unwrap()
            == false
    );
    let grammar = "root ::= scan \"b\"\nscan ::= \"a\" scan | \"b\" scan | \"\"\n";
    let initial = vocab.bind(&plan(&tokenizer, grammar, "")).unwrap();
    for valid in ["b", "ab", "bb", "aabb"] {
        assert!(initial
            .advance(&tokenizer.encode(valid, SpecialTokens::Recognize).unwrap())
            .unwrap()
            .accepting()
            .unwrap());
    }
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
        let initial=vocab.bind(&plan(&tokenizer,&d.grammar,&d.grammar_initial_prefix)).unwrap();
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
    use magnitude_engine::{
        chat::{ChatRequest, PreparedChat, TemplateBundle, TemplateSelection, TemplateVariant},
        generation::{MethodChoice, Options, Sampling, Shaping},
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
    request.json_schema = Some(
        json!({"type":"object","properties":{"message":{"type":"string"}},"required":["message"],"additionalProperties":false}),
    );
    request.reasoning_effort = Some("none".into());
    let prepared =
        PreparedChat::prepare(&bundle, &tokenizer, &request, &TemplateSelection::default())
            .unwrap();
    let mut vocabulary = vocabulary(tokenizer.clone());
    let options = Options {
        max_tokens: 128,
        output_capacity: 16,
        context_limit: 4096,
        vocabulary: 272,
        stop_tokens: tokenizer.stop_tokens().clone(),
        sampling: Sampling::Greedy,
        shaping: Shaping {
            temperature: 0.0,
            ..Default::default()
        },
        seed: 0,
        forced_quantum: 16,
        method: magnitude_engine::generation::MethodChoice::Plain,
    };
    let generation = vocabulary
        .prepare_generation_for_input(
            prepared.input(),
            options.clone(),
            &text_input(prepared.input().tokens.clone()),
        )
        .unwrap();
    assert_eq!(generation.constraint_position(), Some(0));
    assert!(generation.selection_mask().unwrap().is_some());
    let mut interpreted_tokens = prepared.input().tokens.clone();
    interpreted_tokens.push(prepared.input().tokens[0]);
    let interpreted = text_input(interpreted_tokens.clone());
    let interpreted_generation = vocabulary
        .prepare_generation_for_input(prepared.input(), options.clone(), &interpreted)
        .unwrap();
    assert_eq!(interpreted_generation.prompt(), interpreted_tokens);
    let mut bad = prepared.input().clone();
    bad.tokenizer_identity = "another tokenizer".into();
    assert!(vocabulary
        .prepare_generation_for_input(&bad, options.clone(), &text_input(bad.tokens.clone()),)
        .is_err());
    bad = prepared.input().clone();
    bad.constraint.as_mut().unwrap().gbnf = "root ::= missing".into();
    assert!(vocabulary
        .prepare_generation_for_input(&bad, options.clone(), &text_input(bad.tokens.clone()),)
        .is_err());
    let mut bad_options = options.clone();
    bad_options.stop_tokens.clear();
    assert!(vocabulary
        .prepare_generation_for_input(
            prepared.input(),
            bad_options,
            &text_input(prepared.input().tokens.clone()),
        )
        .is_err());

    let mut mtp_options = options;
    mtp_options.method = MethodChoice::Mtp { proposals: 2 };
    let mut mtp_vocabulary = Vocabulary::new(
        tokenizer,
        272,
        CacheLimits {
            entries: 2,
            bytes: 1024 * 1024,
        },
    )
    .unwrap();
    let mtp = mtp_vocabulary
        .prepare_generation_for_input(
            prepared.input(),
            mtp_options,
            &text_input(prepared.input().tokens.clone()),
        )
        .unwrap();
    assert_eq!(mtp.method(), MethodChoice::Mtp { proposals: 2 });
}
