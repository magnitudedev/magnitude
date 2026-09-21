use magnitude_engine::{
    chat::ConstraintPlan,
    generation::{
        constraints::{CacheLimits, Vocabulary},
        grammar::{to_lark, CONVERTER_IDENTITY},
        Constraint,
    },
    inputs::{BpeConfig, ByteBpeTokenizer, PieceKind, SpecialTokens, TokenId},
};
use serde_json::json;
use std::{collections::BTreeSet, sync::Arc};

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
        (include_str!("../../../inference-v3/native/templates/upstream/models/templates/Qwen-Qwen3-0.6B.jinja"),
         "<tool_call>\n{\"name\":\"search\",\"arguments\":{\"query\":\"héllo 世界\"}}\n</tool_call>",
         "<tool_call>\n{\"name\":\"other\",\"arguments\":{\"query\":\"x\"}}\n</tool_call>"),
        (include_str!("../../../inference-v3/native/templates/upstream/models/templates/Qwen3.5-4B.jinja"),
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
fn real_constraint_is_installed_only_after_numerical_commit() {
    use magnitude_engine::{
        generation::{Generation, Options, Readiness, Sampling},
        inputs::InputLayout,
        models::sequence::Advance,
    };
    use std::{cell::Cell, rc::Rc};
    struct Row {
        fail: bool,
        committed: Rc<Cell<bool>>,
    }
    impl Advance for Row {
        fn is_complete(&self) -> bool {
            true
        }
        fn selected(&mut self) -> Result<Option<TokenId>, String> {
            Ok(Some(TokenId(97)))
        }
        fn commit(&mut self) -> Result<(), String> {
            if self.fail {
                return Err("numerical commit failed".into());
            }
            self.committed.set(true);
            Ok(())
        }
    }
    let tokenizer = tokenizer();
    let mut vocab = vocabulary(tokenizer.clone());
    for fail in [true, false] {
        let constraint = vocab
            .bind(&plan(&tokenizer, "root ::= \"ab\" | \"cd\"", ""))
            .unwrap();
        let mut g = Generation::new(
            vec![TokenId(42)],
            InputLayout::new(1, vec![]).unwrap(),
            Options {
                max_tokens: 8,
                output_capacity: 8,
                context_limit: 32,
                vocabulary: 272,
                stop_tokens: tokenizer.stop_tokens().clone(),
                sampling: Sampling::Greedy,
                seed: 7,
                forced_quantum: 0,
            },
            Some(Box::new(constraint)),
        )
        .unwrap();
        let before = g.selection_mask().unwrap().unwrap();
        assert!(allowed(&before, 97));
        assert!(allowed(&before, 99));
        let Readiness::Ready(proposal) = g.ready(1).unwrap() else {
            panic!("expected prefill")
        };
        assert!(proposal.needs_sample());
        let committed = Rc::new(Cell::new(false));
        g.attach(
            proposal,
            Box::new(Row {
                fail,
                committed: committed.clone(),
            }),
        )
        .unwrap();
        assert_eq!(g.reconcile().is_err(), fail);
        assert_eq!(committed.get(), !fail);
        assert_eq!(g.constraint_position(), Some(usize::from(!fail)));
        let after = g.selection_mask().unwrap().unwrap();
        if fail {
            assert_eq!(before, after);
            assert!(g.generated().is_empty());
        } else {
            assert!(!allowed(&after, 97));
            assert!(allowed(&after, 98));
            assert_eq!(g.generated(), &[TokenId(97)]);
        }
    }
}

#[test]
fn host_preparation_binds_before_generation_and_rejects_identity_mismatches() {
    use magnitude_engine::{
        chat::{ChatRequest, PreparedChat, TemplateBundle, TemplateSelection, TemplateVariant},
        generation::{Options, Sampling},
    };
    let tokenizer = tokenizer();
    let bundle = TemplateBundle::new(vec![TemplateVariant {
        name: "default".into(),
        source: include_str!("../../../inference-v3/native/templates/upstream/models/templates/Qwen-Qwen3-0.6B.jinja").into(),
        provenance: "fixture".into(),
    }], "default".into(), Default::default()).unwrap();
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
        seed: 0,
        forced_quantum: 16,
    };
    let generation = vocabulary
        .prepare_generation(prepared.input(), options.clone())
        .unwrap();
    assert_eq!(generation.constraint_position(), Some(0));
    assert!(generation.selection_mask().unwrap().is_some());
    let mut bad = prepared.input().clone();
    bad.tokenizer_identity = "another tokenizer".into();
    assert!(vocabulary
        .prepare_generation(&bad, options.clone())
        .is_err());
    bad = prepared.input().clone();
    bad.constraint.as_mut().unwrap().gbnf = "root ::= missing".into();
    assert!(vocabulary
        .prepare_generation(&bad, options.clone())
        .is_err());
    let mut bad_options = options;
    bad_options.stop_tokens.clear();
    assert!(vocabulary
        .prepare_generation(prepared.input(), bad_options)
        .is_err());
}

#[test]
#[ignore = "requires a Metal device"]
fn generation_checkpoint_forks_matcher_output_and_numerical_continuation_together() {
    use magnitude_engine::{
        generation::{FinishReason, Generation, Options, Readiness, Sampling},
        inputs::InputLayout,
        models::sequence::OwnedSequence,
        state::{ComponentSpec, StateStore},
    };
    use seismic::{BackendName, DType, DeviceCatalog};
    use std::rc::Rc;
    fn advance(g: &mut Generation, sequence: &OwnedSequence, token: u32) {
        let Readiness::Ready(proposal) = g.ready(8).unwrap() else {
            panic!("ready")
        };
        let row = sequence
            .prepare_completed(proposal.position(), proposal.tokens().len(), |state| {
                let mut next = state.begin(proposal.tokens().len())?;
                next.execute(|b| {
                    let mut value = b.following[0].clone();
                    value
                        .write_from_host(&(token as f32).to_le_bytes())
                        .map_err(Into::into)
                })?;
                next.commit()?;
                Ok(Some(TokenId(token)))
            })
            .unwrap();
        g.attach(proposal, row).unwrap();
        g.reconcile().unwrap();
    }
    fn value(sequence: &OwnedSequence) -> f32 {
        let state = sequence.checkpoint().unwrap().fork();
        let bytes = state.values()[0].read_to_host().unwrap();
        f32::from_le_bytes(bytes.try_into().unwrap())
    }
    let tokenizer = tokenizer();
    let grammar = vocabulary(tokenizer.clone())
        .bind(&plan(&tokenizer, "root ::= \"ab\" (\"c\" | \"d\")", ""))
        .unwrap();
    let mut generation = Generation::new(
        vec![TokenId(1), TokenId(2)],
        InputLayout::new(2, vec![]).unwrap(),
        Options {
            max_tokens: 8,
            output_capacity: 4,
            context_limit: 16,
            vocabulary: 272,
            stop_tokens: tokenizer.stop_tokens().clone(),
            sampling: Sampling::Categorical,
            seed: 42,
            forced_quantum: 0,
        },
        Some(Box::new(grammar)),
    )
    .unwrap();
    let store = StateStore::new(
        Rc::new(
            DeviceCatalog::discover()
                .unwrap()
                .open_backend(BackendName::Metal)
                .unwrap(),
        ),
        16,
        64,
        vec![ComponentSpec {
            shape: vec![4],
            dtype: DType::F32,
        }],
        vec![ComponentSpec {
            shape: vec![1],
            dtype: DType::F32,
        }],
    )
    .unwrap();
    let sequence = OwnedSequence::new(store.create().unwrap());
    advance(&mut generation, &sequence, 97);
    assert_eq!(generation.take(1).unwrap()[0].index, 0);
    advance(&mut generation, &sequence, 98);
    let checkpoint = generation.checkpoint(&sequence).unwrap();
    assert_eq!(checkpoint.position(), 3);
    let (mut left, left_state) = checkpoint.fork().unwrap();
    let (mut right, right_state) = checkpoint.fork().unwrap();
    drop(generation);
    drop(sequence);
    assert_eq!(left.take(1).unwrap()[0].index, 1);
    assert_eq!(right.output_len(), 1);
    assert_eq!(right.take(1).unwrap()[0].token, TokenId(98));
    advance(&mut left, &left_state, 99);
    assert_eq!(value(&left_state), 99.0);
    assert_eq!(value(&right_state), 98.0);
    let mask = right.selection_mask().unwrap().unwrap();
    assert!(allowed(&mask, 99) && allowed(&mask, 100));
    advance(&mut right, &right_state, 100);
    assert_eq!(left.take(1).unwrap()[0].index, 2);
    assert_eq!(right.take(1).unwrap()[0].token, TokenId(100));
    assert_eq!(left.generated(), &[TokenId(97), TokenId(98), TokenId(99)]);
    assert_eq!(right.generated(), &[TokenId(97), TokenId(98), TokenId(100)]);
    advance(&mut left, &left_state, 256);
    assert_eq!(left.finish_reason(), Some(FinishReason::Stop));
    let terminal = left.checkpoint(&left_state).unwrap();
    let (stopped, _) = terminal.fork().unwrap();
    assert_eq!(stopped.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(stopped.constraint_position(), Some(4));
    // The original immutable checkpoint still has the old output and matcher.
    let (mut again, state) = checkpoint.fork().unwrap();
    assert_eq!(again.take(1).unwrap()[0].index, 1);
    assert_eq!(again.constraint_position(), Some(2));
    assert_eq!(value(&state), 98.0);
}
