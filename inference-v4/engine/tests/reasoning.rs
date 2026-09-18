use magnitude_templates::Template;
use seismic_engine::chat::{
    reasoning::{inspect_reasoning, ReasoningProfile},
    ChatRequest, TemplateBundle, TemplateSelection, TemplateVariant,
};
use serde_json::{json, Value};

fn fixtures() -> Value {
    serde_json::from_str(include_str!(
        "../../../inference-v3/tests/templates/reasoning-fixtures.json"
    ))
    .unwrap()
}

#[test]
fn profiles_preserve_v3_behavior_classification_and_omission() {
    let fixtures = fixtures();
    for (name, efforts) in [
        ("BASIC", vec!["none"]),
        ("TOGGLE", vec!["none", "high"]),
        ("FIXED", vec!["high"]),
        ("THINKING_BOOL", vec!["none", "high"]),
        ("THINKING_MODE", vec!["none", "adaptive", "high"]),
        ("EFFORT_TOGGLE", vec!["none", "high"]),
        ("EFFORT_NONE_MATCHES_LOW", vec!["none", "low", "high"]),
        ("CLOSED_EFFORT", vec!["none", "low", "medium", "high"]),
        ("QWEN_3_8_EFFORT", vec!["none", "low", "medium", "xhigh"]),
        ("REVERSE_EFFORT_ALIAS", vec!["low", "medium", "high"]),
        ("ONE_ENABLED_EFFORT_BEHAVIOR", vec!["none", "max"]),
        ("SHARED_FALLBACK_EFFORT", vec!["none", "low", "high"]),
        ("NAMED_SHARED_FALLBACK_EFFORT", vec!["none", "high", "max"]),
        ("OPEN_EFFORT", vec!["none", "high"]),
    ] {
        let template =
            Template::new(fixtures[name].as_str().unwrap(), &Default::default()).unwrap();
        let profile = inspect_reasoning(&template, &Default::default()).unwrap();
        assert_eq!(
            profile
                .mappings
                .iter()
                .map(|m| m.effort.as_str())
                .collect::<Vec<_>>(),
            efforts,
            "{name}"
        );
        assert!(profile.resolve(None).unwrap().is_empty());
        assert_eq!(
            serde_json::from_slice::<ReasoningProfile>(&serde_json::to_vec(&profile).unwrap())
                .unwrap(),
            profile
        );
        if name == "QWEN_3_8_EFFORT" {
            assert_eq!(profile.default_effort.as_deref(), Some("xhigh"));
            assert_eq!(
                profile.resolve(Some("high")).unwrap(),
                profile.resolve(Some("xhigh")).unwrap()
            );
        }
    }
}

#[test]
fn independent_options_change_profile_and_fingerprint_deterministically() {
    let fixtures = fixtures();
    let source = format!(
        "{{% if extra_levels %}}{}{{% else %}}{}{{% endif %}}",
        fixtures["CLOSED_EFFORT"].as_str().unwrap(),
        fixtures["TOGGLE"].as_str().unwrap()
    );
    let template = Template::new(&source, &Default::default()).unwrap();
    let plain = inspect_reasoning(&template, &Default::default()).unwrap();
    let context = json!({"extra_levels":true,"nested":{"b":1,"a":2}})
        .as_object()
        .unwrap()
        .clone();
    let extended = inspect_reasoning(&template, &context).unwrap();
    assert_eq!(plain.mappings.len(), 2);
    assert_eq!(extended.mappings.len(), 4);
    assert_eq!(plain.template_identity, extended.template_identity);
    assert_ne!(plain.fingerprint(), extended.fingerprint());
    let reordered = json!({"nested":{"a":2,"b":1},"extra_levels":true})
        .as_object()
        .unwrap()
        .clone();
    assert_eq!(
        extended.fingerprint(),
        inspect_reasoning(&template, &reordered)
            .unwrap()
            .fingerprint()
    );
    assert!(inspect_reasoning(&template, json!({"thinking":false}).as_object().unwrap()).is_err());
}

#[test]
fn request_controls_preserve_authored_default_and_reject_conflicts() {
    let source = fixtures()["TOGGLE"].as_str().unwrap().to_owned();
    let bundle = TemplateBundle::new(
        vec![TemplateVariant {
            name: "default".into(),
            source,
            provenance: "test".into(),
        }],
        "default".into(),
        Default::default(),
    )
    .unwrap();
    let mut request = ChatRequest::new(vec![json!({"role":"user","content":"hello"})], 0);
    let default = bundle
        .prepare(&request, &TemplateSelection::default())
        .unwrap();
    assert!(!default.description().prompt.contains("<think>"));
    request.reasoning_effort = Some("high".into());
    assert!(bundle
        .prepare(&request, &TemplateSelection::default())
        .unwrap()
        .description()
        .prompt
        .contains("<think>"));
    assert!(request.template_arguments.is_empty());
    request.reasoning_effort = Some("medium".into());
    assert!(bundle
        .prepare(&request, &TemplateSelection::default())
        .err()
        .unwrap()
        .contains("unsupported reasoning effort"));
    request
        .template_arguments
        .insert("thinking".into(), json!(false));
    assert!(bundle
        .prepare(&request, &TemplateSelection::default())
        .err()
        .unwrap()
        .contains("conflicts"));
}

#[test]
fn profile_cache_is_bounded_and_separates_source_and_options() {
    use seismic_engine::chat::reasoning::ProfileCache;
    use std::rc::Rc;
    let fixtures = fixtures();
    let template =
        Template::new(fixtures["TOGGLE"].as_str().unwrap(), &Default::default()).unwrap();
    let mut cache = ProfileCache::new(2, 8192);
    let first = cache.inspect(&template, &Default::default()).unwrap();
    let same = cache.inspect(&template, &Default::default()).unwrap();
    assert!(Rc::ptr_eq(&first, &same));
    assert_eq!(cache.hits(), 1);
    for n in 0..3 {
        cache
            .inspect(&template, json!({"independent":n}).as_object().unwrap())
            .unwrap();
    }
    assert_eq!(cache.cached_entries(), 2);
    assert!(cache.cached_bytes() <= 8192);
    assert!(!Rc::ptr_eq(
        &first,
        &cache.inspect(&template, &Default::default()).unwrap()
    ));
    let other = Template::new(fixtures["FIXED"].as_str().unwrap(), &Default::default()).unwrap();
    assert_ne!(
        first.template_identity,
        cache
            .inspect(&other, &Default::default())
            .unwrap()
            .template_identity
    );
    let mut disabled = ProfileCache::new(16, 1);
    disabled.inspect(&template, &Default::default()).unwrap();
    assert_eq!(disabled.cached_entries(), 0);
    assert_eq!(disabled.cached_bytes(), 0);
}
