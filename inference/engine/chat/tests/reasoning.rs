use magnitude_chat::reasoning::{
    inspect_reasoning, normalize_effort, EffortDomain, EffortMapping, ReasoningError,
    ReasoningIntent, ReasoningProfile, DETECTOR,
};
use magnitude_templates::Template;
use serde_json::json;
use std::collections::BTreeMap;

fn fixtures() -> BTreeMap<String, String> {
    serde_json::from_str(include_str!("assets/reasoning-fixtures.json")).unwrap()
}

fn profile_of(name: &str) -> ReasoningProfile {
    let template = Template::new(&fixtures()[name], &Default::default()).unwrap();
    inspect_reasoning(&template, &Default::default()).unwrap()
}

#[test]
fn behavioral_reasoning_discovery_preserves_the_fourteen_classifications() {
    let expected = [
        ("BASIC", &["none"][..]),
        ("TOGGLE", &["none", "high"]),
        ("FIXED", &["high"]),
        ("THINKING_BOOL", &["none", "high"]),
        ("THINKING_MODE", &["none", "adaptive", "high"]),
        ("EFFORT_TOGGLE", &["none", "high"]),
        ("EFFORT_NONE_MATCHES_LOW", &["none", "low", "high"]),
        ("CLOSED_EFFORT", &["none", "low", "medium", "high"]),
        ("QWEN_3_8_EFFORT", &["none", "low", "medium", "xhigh"]),
        ("REVERSE_EFFORT_ALIAS", &["low", "medium", "high"]),
        ("ONE_ENABLED_EFFORT_BEHAVIOR", &["none", "max"]),
        ("SHARED_FALLBACK_EFFORT", &["none", "low", "high"]),
        ("NAMED_SHARED_FALLBACK_EFFORT", &["none", "high", "max"]),
        ("OPEN_EFFORT", &["none", "high"]),
    ];
    for (name, efforts) in expected {
        let profile = profile_of(name);
        assert_eq!(
            profile
                .mappings
                .iter()
                .map(|mapping| mapping.effort.as_str())
                .collect::<Vec<_>>(),
            efforts,
            "classification changed for {name}"
        );
        let default = profile.resolve(&ReasoningIntent::ModelDefault).unwrap();
        assert!(default.controls.is_empty(), "the default renders as the template does");
        let encoded = serde_json::to_vec(&profile).unwrap();
        assert_eq!(
            serde_json::from_slice::<ReasoningProfile>(&encoded).unwrap(),
            profile
        );
    }
}

fn effort(value: &str) -> ReasoningIntent {
    ReasoningIntent::Effort {
        effort: value.into(),
    }
}

#[test]
fn recognized_effort_alias_resolves_without_inventing_a_level() {
    let profile = profile_of("QWEN_3_8_EFFORT");
    assert_eq!(profile.default_effort.as_deref(), Some("xhigh"));
    assert_eq!(
        profile.resolve(&effort("high")).unwrap(),
        profile.resolve(&effort("xhigh")).unwrap()
    );
    assert!(!profile
        .mappings
        .iter()
        .any(|mapping| mapping.effort == "high"));
}

/// A synthetic profile whose mappings all render distinctly.
fn synthetic(efforts: &[&str], default: &str) -> ReasoningProfile {
    ReasoningProfile {
        detector: DETECTOR.into(),
        template_identity: "fixture".into(),
        option_context: "{}".into(),
        default_effort: Some(default.into()),
        mappings: efforts
            .iter()
            .map(|effort| EffortMapping {
                effort: (*effort).into(),
                controls: BTreeMap::from([("reasoning_effort".into(), json!(effort))]),
                aliases: Vec::new(),
            })
            .collect(),
        baseline_shapes: vec![true],
        effort_domain: EffortDomain::Closed,
        supports_reasoning_output: Some(true),
        supports_preserve_reasoning: false,
    }
}

#[test]
fn exact_efforts_are_preserved_and_unsupported_ordinals_round_up_then_clamp() {
    for (efforts, default, requested, expected) in [
        (vec!["low", "xhigh"], "low", "low", "low"),
        (vec!["low", "xhigh"], "low", "medium", "xhigh"),
        (vec!["low", "medium", "xhigh"], "medium", "high", "xhigh"),
        (vec!["low", "high"], "high", "xhigh", "high"),
        (vec!["low", "xhigh"], "xhigh", "max", "xhigh"),
        (vec!["high"], "high", "medium", "high"),
        (vec!["adaptive"], "adaptive", "medium", "adaptive"),
        (vec!["none", "low", "high"], "high", "extra-high", "high"),
    ] {
        let resolved = synthetic(&efforts, default)
            .resolve(&effort(requested))
            .unwrap();
        assert_eq!(
            resolved.effort.as_deref(),
            Some(expected),
            "{requested} over {efforts:?}"
        );
        assert_eq!(resolved.controls["reasoning_effort"], json!(expected));
    }
}

#[test]
fn named_modes_fall_back_to_the_enabled_default() {
    let resolved = synthetic(&["none", "low", "high"], "high")
        .resolve(&effort("adaptive"))
        .unwrap();
    assert_eq!(resolved.effort.as_deref(), Some("high"));
    // A disabled default is not an enabled fallback.
    let resolved = synthetic(&["none", "low", "high"], "none")
        .resolve(&effort("adaptive"))
        .unwrap();
    assert_eq!(resolved.effort.as_deref(), Some("low"));
    let resolved = synthetic(&["none", "medium"], "none")
        .resolve(&ReasoningIntent::Enabled)
        .unwrap();
    assert_eq!(resolved.effort.as_deref(), Some("medium"));
}

#[test]
fn unsupported_disable_is_an_error() {
    let profile = synthetic(&["low", "high"], "high");
    for intent in [effort("none"), effort("off"), ReasoningIntent::Disabled] {
        assert!(
            matches!(
                profile.resolve(&intent),
                Err(ReasoningError::CannotDisable { .. })
            ),
            "{intent:?}"
        );
    }
    let fixed = profile_of("FIXED");
    assert!(fixed.resolve(&ReasoningIntent::Disabled).is_err());
    let toggle = profile_of("TOGGLE");
    assert_eq!(
        toggle.resolve(&ReasoningIntent::Disabled).unwrap().controls,
        BTreeMap::from([("enable_thinking".into(), json!(false))])
    );
}

#[test]
fn a_model_without_enabled_reasoning_cannot_enable_it() {
    let basic = profile_of("BASIC");
    assert!(matches!(
        basic.resolve(&ReasoningIntent::Enabled),
        Err(ReasoningError::NoEnabledEffort { .. })
    ));
    assert!(matches!(
        basic.resolve(&effort("high")),
        Err(ReasoningError::UnsupportedEffort { .. })
    ));
}

/// Real LFM2.5 templates are fixed reasoning: 2.6B opens `<think>` in its
/// generation prompt, while the 1.2B Thinking and Instruct finetunes share one
/// byte-identical template that leaves the block to the model and strips
/// `</think>` from history.
#[test]
fn lfm2_5_templates_are_fixed_reasoning() {
    for name in ["LFM2_5_THINKING", "LFM2_5_1_2B_THINKING", "LFM2_5_INSTRUCT"] {
        let profile = profile_of(name);
        assert_eq!(
            profile
                .mappings
                .iter()
                .map(|mapping| mapping.effort.as_str())
                .collect::<Vec<_>>(),
            ["high"],
            "{name}"
        );
        assert_eq!(profile.default_effort.as_deref(), Some("high"), "{name}");
        assert_eq!(profile.supports_reasoning_output, Some(true), "{name}");
        assert!(
            matches!(
                profile.resolve(&effort("none")),
                Err(ReasoningError::CannotDisable { .. })
            ),
            "{name}"
        );
    }
    assert_eq!(
        fixtures()["LFM2_5_1_2B_THINKING"],
        fixtures()["LFM2_5_INSTRUCT"]
    );
}

/// The real Llama 3.2 1B Instruct template (bartowski/LM Studio GGUF) has no
/// reasoning control, so it is the single disabled option `none`.
#[test]
fn llama_3_2_instruct_normalizes_to_none() {
    let profile = profile_of("LLAMA_3_2_INSTRUCT");
    assert_eq!(
        profile
            .mappings
            .iter()
            .map(|mapping| mapping.effort.as_str())
            .collect::<Vec<_>>(),
        ["none"]
    );
    assert_eq!(profile.default_effort.as_deref(), Some("none"));
    assert_eq!(profile.supports_reasoning_output, None);
    let disabled = profile.resolve(&effort("none")).unwrap();
    assert_eq!(disabled.effort.as_deref(), Some("none"));
    assert!(disabled.controls.is_empty());
    assert!(matches!(
        profile.resolve(&effort("high")),
        Err(ReasoningError::UnsupportedEffort { .. })
    ));
}

#[test]
fn disabled_and_extra_high_spellings_normalize() {
    for spelling in ["none", "off", "no_think", "disabled"] {
        assert_eq!(normalize_effort(spelling), Some("none"));
    }
    for spelling in ["xhigh", "extra_high", "extra-high", "very_high"] {
        assert_eq!(normalize_effort(spelling), Some("xhigh"));
    }
    assert_eq!(normalize_effort("adaptive"), Some("adaptive"));
    assert_eq!(normalize_effort("ultra"), None);
}

fn asset_profile(source: &str) -> ReasoningProfile {
    let template = Template::new(source, &Default::default()).unwrap();
    inspect_reasoning(&template, &Default::default()).unwrap()
}

/// Turning reasoning off counts only when the generation prompt no longer
/// leaves the reply inside a reasoning block. The DavidAU LFM2.5 8B-A1B
/// TBrilliance template swaps its system prompt for `enable_thinking=false`
/// and for `reasoning_effort="off"` but always ends by opening `<think>`, so
/// it keeps its native efforts and cannot be disabled. Qwen 3.5 closes the
/// block empty when reasoning is off, which does disable it.
#[test]
fn an_off_setting_that_still_opens_reasoning_does_not_disable_it() {
    let tbrilliance = asset_profile(include_str!(
        "../../templates/tests/assets/LFM2.5-8B-A1B-Q3.8-TBrilliance.jinja"
    ));
    assert_eq!(
        tbrilliance
            .mappings
            .iter()
            .map(|mapping| mapping.effort.as_str())
            .collect::<Vec<_>>(),
        ["low", "medium", "high"]
    );
    assert_eq!(tbrilliance.default_effort.as_deref(), Some("high"));
    for intent in [effort("none"), ReasoningIntent::Disabled] {
        assert!(
            matches!(
                tbrilliance.resolve(&intent),
                Err(ReasoningError::CannotDisable { .. })
            ),
            "{intent:?}"
        );
    }
    let qwen = asset_profile(include_str!("../../templates/tests/assets/Qwen3.5-4B.jinja"));
    assert_eq!(
        qwen.resolve(&ReasoningIntent::Disabled).unwrap().controls,
        BTreeMap::from([("enable_thinking".into(), json!(false))])
    );
}

#[test]
fn muse_glimmer_reasoning_can_be_disabled() {
    let template = Template::new(
        include_str!("../../templates/tests/assets/Muse-Glimmer-30B.jinja"),
        &Default::default(),
    )
    .unwrap();
    let profile = inspect_reasoning(&template, &Default::default()).unwrap();
    assert_eq!(
        profile
            .mappings
            .iter()
            .map(|mapping| mapping.effort.as_str())
            .collect::<Vec<_>>(),
        ["none", "high"]
    );
    assert_eq!(profile.default_effort.as_deref(), Some("high"));
    let disabled = profile.resolve(&ReasoningIntent::Disabled).unwrap();
    assert_eq!(
        disabled.controls,
        BTreeMap::from([("enable_thinking".to_string(), json!(false))])
    );
}

/// The renderer enables thinking unless the caller turns it off, so a toggle's
/// enabled side is the default even when a template's own Jinja falls back to
/// disabled, as the real Gemma 4 template does with `default(false)`.
#[test]
fn an_omitted_effort_selects_the_enabled_side_of_a_toggle() {
    let gemma = asset_profile(include_str!(
        "../../templates/tests/assets/gemma-4-12B-it.jinja"
    ));
    assert_eq!(
        gemma
            .mappings
            .iter()
            .map(|mapping| mapping.effort.as_str())
            .collect::<Vec<_>>(),
        ["none", "high"]
    );
    assert_eq!(gemma.default_effort.as_deref(), Some("high"));
    assert_eq!(
        gemma.resolve(&ReasoningIntent::Disabled).unwrap().controls,
        BTreeMap::from([("enable_thinking".into(), json!(false))])
    );
    for (name, default) in [
        ("BASIC", Some("none")),
        ("TOGGLE", Some("high")),
        ("FIXED", Some("high")),
        ("THINKING_BOOL", Some("none")),
        ("THINKING_MODE", Some("adaptive")),
        ("EFFORT_TOGGLE", Some("high")),
        ("EFFORT_NONE_MATCHES_LOW", Some("high")),
        ("CLOSED_EFFORT", Some("high")),
        ("QWEN_3_8_EFFORT", Some("xhigh")),
        ("REVERSE_EFFORT_ALIAS", Some("high")),
        ("ONE_ENABLED_EFFORT_BEHAVIOR", Some("max")),
        ("SHARED_FALLBACK_EFFORT", Some("high")),
        ("NAMED_SHARED_FALLBACK_EFFORT", Some("max")),
        ("UNNAMED_DEFAULT_EFFORT", None),
        ("OPEN_EFFORT", Some("high")),
    ] {
        assert_eq!(profile_of(name).default_effort.as_deref(), default, "{name}");
    }
}
