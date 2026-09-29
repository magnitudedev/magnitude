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
