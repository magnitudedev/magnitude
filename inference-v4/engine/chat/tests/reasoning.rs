use magnitude_chat::reasoning::{inspect_reasoning, ReasoningProfile};
use magnitude_templates::Template;
use std::collections::BTreeMap;

#[test]
fn behavioral_reasoning_discovery_preserves_the_fourteen_classifications() {
    let fixtures: BTreeMap<String, String> =
        serde_json::from_str(include_str!("assets/reasoning-fixtures.json")).unwrap();
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
        let template = Template::new(&fixtures[name], &Default::default()).unwrap();
        let profile = inspect_reasoning(&template, &Default::default()).unwrap();
        assert_eq!(
            profile
                .mappings
                .iter()
                .map(|mapping| mapping.effort.as_str())
                .collect::<Vec<_>>(),
            efforts,
            "classification changed for {name}"
        );
        assert!(profile.resolve(None).unwrap().is_empty());
        let encoded = serde_json::to_vec(&profile).unwrap();
        assert_eq!(
            serde_json::from_slice::<ReasoningProfile>(&encoded).unwrap(),
            profile
        );
    }
}

#[test]
fn recognized_effort_alias_resolves_without_inventing_a_level() {
    let fixtures: BTreeMap<String, String> =
        serde_json::from_str(include_str!("assets/reasoning-fixtures.json")).unwrap();
    let template = Template::new(&fixtures["QWEN_3_8_EFFORT"], &Default::default()).unwrap();
    let profile = inspect_reasoning(&template, &Default::default()).unwrap();
    assert_eq!(profile.default_effort.as_deref(), Some("xhigh"));
    assert_eq!(
        profile.resolve(Some("high")),
        profile.resolve(Some("xhigh"))
    );
    assert!(!profile
        .mappings
        .iter()
        .any(|mapping| mapping.effort == "high"));
}
