//! Planning sizes every attention call's scratch over the configurations its
//! Metal declaration admits (`NativeImplementation::scratch_bound`). Those
//! declarations list many values per parameter, entry-wide and per launch,
//! so a bound that walks the product of their domains takes minutes at
//! DeepSeek V4.1's shapes; it must take a moment.

use seismic_lang::bundle::decode_checked_bundle;
use seismic_lang::entry::ElementBindings;
use seismic_lang::registry::{self, BackendName};
use seismic_lang::types::DType;
use std::time::{Duration, Instant};

static BUNDLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kernels.seismicbundle"));

/// DeepSeek V4.1 Flash's window attention by the family's static names
/// (`AttentionGeometry::mix_statics`): 64 query heads over one key/value
/// head 512 wide (32 adjacent trailing rotary pairs, 448 other dims), one
/// fresh layer, a key norm and no query or value norm, sinks, the attended
/// row turned back, and the stored key as the value.
const V41_MIX: [(&str, u64); 15] = [
    ("KV", 1),
    ("G", 64),
    ("P", 32),
    ("S", 448),
    ("I", 0),
    ("U", 0),
    ("F", 1),
    ("NQ", 0),
    ("N", 1),
    ("NV", 0),
    ("SK", 1),
    ("RA", 1),
    ("RT", 1),
    ("RO", 1),
    ("VK", 1),
];

/// The statics `entry` is held to at DeepSeek V4.1's shapes: its
/// projections over a 4096-wide hidden state (query rows 64 * 512, one key
/// head and no value or gate rows of its own; the output of 64 heads 512
/// wide), the mix entries' [`V41_MIX`]. The affine K8/V4 entries hold
/// values of their own (`VK = 0`), which V4.1 does not, so they are held to
/// its head geometry with projected values.
fn v41_statics(entry: &str) -> Vec<(&'static str, u64)> {
    match entry {
        "attention_project" => vec![("D", 4096), ("Q", 64 * 512), ("GR", 0), ("K", 512), ("V", 0)],
        "attention_output" => vec![("D", 4096), ("Q", 64), ("W", 512)],
        entry if entry.ends_with("_k8v4") => V41_MIX
            .iter()
            .map(|&(name, value)| (name, if name == "VK" { 0 } else { value }))
            .collect(),
        _ => V41_MIX.to_vec(),
    }
}

#[test]
fn metal_attention_scratch_bounds_at_deepseek_v41_take_a_moment() {
    let module = decode_checked_bundle(BUNDLE).expect("the embedded kernel bundle decodes");
    let mut admitted = Vec::new();
    let mut spent = Duration::ZERO;
    for entry in module.entries() {
        if !entry.name.starts_with("attention_") {
            continue;
        }
        if module.native_implementation(entry.id, BackendName::Metal).is_none() {
            continue;
        }
        let native = [DType::BF16, DType::F16, DType::F32]
            .into_iter()
            .find_map(|dtype| {
                let bindings = entry
                    .element_parameters
                    .iter()
                    .fold(ElementBindings::new(), |bindings, name| {
                        bindings.bind(name, registry::dense(dtype))
                    });
                module.native_implementation_at(entry.id, BackendName::Metal, &bindings)
            })
            .unwrap_or_else(|| panic!("`{}` forms at dense elements on Metal", entry.name));
        let fixed = v41_statics(&entry.name)
            .into_iter()
            .filter(|(name, _)| native.statics.iter().any(|candidate| candidate == name))
            .collect::<Vec<_>>();
        // Metal's decode holds no form of 64 heads 512 wide (the prefill
        // entry attends V4.1's decode rows), so not every entry admits them.
        let Some(statics) = native
            .search_statics(&fixed)
            .unwrap_or_else(|error| panic!("`{}`: {error}", entry.name))
        else {
            eprintln!("`{}` admits no statics at {fixed:?}", entry.name);
            continue;
        };
        admitted.push(entry.name.clone());
        // A decode row and a prefill batch, over a long history.
        for rows in [1, 4096] {
            let dimension = |name: &str| match name {
                "M" => Some(rows),
                name => statics.static_value(name).or(Some(4096)),
            };
            let started = Instant::now();
            let bound = native.scratch_bound(&dimension);
            let elapsed = started.elapsed();
            eprintln!("`{}` at M = {rows}: {bound:?} in {elapsed:?}", entry.name);
            bound.unwrap_or_else(|error| panic!("`{}` at M = {rows}: {error}", entry.name));
            spent += elapsed;
        }
    }
    // The entries a V4.1 window layer plans with on Metal.
    for entry in ["attention_project", "attention_prefill", "attention_append_dense", "attention_output"] {
        assert!(admitted.iter().any(|name| name == entry), "`{entry}` is bounded at V4.1's statics");
    }
    assert!(spent < Duration::from_secs(1), "bounding took {spent:?}");
}
