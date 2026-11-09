//! A checked module decoded from its bundle forms byte-identical native
//! sources to the module the checker produced from source text.
//!
//! For every entry of the engine kernel library with a Metal, CUDA or Vulkan
//! native implementation, under every admissible binding of its element
//! parameters to the representations the Qwen path stores (dense f32, f16
//! and bf16; q4k, q5k, q6k and q8g32s in `rows16`), the `LogicalEntry` is
//! built from both modules and the formation source is rendered for an
//! admissible specialization: the MSL the Metal pipeline compiles, the CUDA
//! source NVRTC compiles to PTX, and the GLSL the Vulkan pipeline compiles.
//! Every asset renders the module constants it names as the per-constant
//! scan rendered them, in one pass.

use super::abi::{
    constant_macros, native_macro, render_source, vulkan::VulkanFeatures, CudaFeatures, Dialect,
    MetalFeatures,
};
use seismic_lang::bundle::{decode_checked_bundle, encode_checked_bundle};
use seismic_lang::checked::{
    CheckedModule, ElementParameter, NativeAsset, NativeImplementation, NativeSpecialization,
};
use seismic_lang::entry::ElementBindings;
use seismic_lang::ids::RepresentationId;
use seismic_lang::registry::{self, BackendName, Layout};
use std::collections::BTreeMap;
use std::time::Instant;

const VULKAN: VulkanFeatures = VulkanFeatures {
    subgroup_lanes: 32,
    float16: true,
    matrix: true,
    wide_accumulators: true,
    mixed_dot: true,
    f32_atomic_add: true,
    shared_int64_atomics: false,
};

/// At most this many element bindings per entry.
const BINDINGS_PER_ENTRY: usize = 24;

fn qwen_representations() -> Vec<RepresentationId> {
    let mut representations: Vec<_> = ["f32", "f16", "bf16"]
        .into_iter()
        .map(|name| registry::representation(name).expect("dense representation"))
        .collect();
    for packed in ["q4k", "q5k", "q6k", "q8g32s"] {
        representations.push(registry::storage(packed, Layout::Rows16).expect("rows16 storage"));
    }
    representations
}

/// Admissible bindings of `parameters`, in a fixed order, at most
/// [`BINDINGS_PER_ENTRY`].
fn bindings(
    parameters: &[ElementParameter],
    representations: &[RepresentationId],
    admit: impl Fn(&ElementBindings) -> bool,
) -> Vec<ElementBindings> {
    let mut found = Vec::new();
    let mut choice = vec![0usize; parameters.len()];
    loop {
        let candidate = parameters
            .iter()
            .zip(&choice)
            .fold(ElementBindings::new(), |bound, (parameter, index)| {
                bound.bind(&parameter.name, representations[*index])
            });
        if admit(&candidate) {
            found.push(candidate);
            if found.len() == BINDINGS_PER_ENTRY {
                return found;
            }
        }
        // Next combination, last parameter fastest.
        let mut position = parameters.len();
        loop {
            if position == 0 {
                return found;
            }
            position -= 1;
            choice[position] += 1;
            if choice[position] < representations.len() {
                break;
            }
            choice[position] = 0;
        }
    }
}

/// A deterministic admissible specialization of `native`.
fn specialization(native: &NativeImplementation) -> NativeSpecialization {
    native
        .device_independent()
        .unwrap()
        .default_specialization(&statics(native).unwrap_or_else(|| {
            panic!("no admissible specialization of `{}` found", native.source_path)
        }))
        .expect("the statics admit the default")
}

/// Deterministic static values at which `native` has an admissible default,
/// when the search finds any. Each draw takes the statics in declaration
/// order, each from the values its `where` condition can hold at beside
/// those drawn before it ([`NativeImplementation::may_admit`], the rest
/// anywhere in [`VALUES`]'s range): a draw lies within every bound the
/// declaration states, and only what the condition's bounds cannot decide
/// (a remainder, a joint condition) is left to the default's walk.
fn statics(native: &NativeImplementation) -> Option<NativeSpecialization> {
    let range = (VALUES[0], VALUES[VALUES.len() - 1]);
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    'draw: for _ in 0..20_000 {
        let mut statics = NativeSpecialization::new();
        for name in &native.statics {
            let admitted = VALUES
                .into_iter()
                .filter(|&value| {
                    native.may_admit(&|dimension| {
                        if dimension == name.as_str() {
                            Some((value, value))
                        } else {
                            Some(statics.static_value(dimension).map_or(range, |drawn| (drawn, drawn)))
                        }
                    })
                })
                .collect::<Vec<_>>();
            if admitted.is_empty() {
                continue 'draw;
            }
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            statics = statics.with_static(name.as_str(), admitted[(state >> 33) as usize % admitted.len()]);
        }
        if native.device_independent().unwrap().default_specialization(&statics).is_ok() {
            return Some(statics);
        }
    }
    None
}

/// The values a static dimension is drawn from: zero (an absent operand or
/// a flag off) and the sizes kernels declare.
const VALUES: [u64; 15] = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 2560, 4096,
];

/// Slow: checks the whole engine kernel library. Run it in release with
/// `--ignored`.
#[test]
#[ignore]
fn decoded_kernel_bundle_forms_identical_native_sources() {
    let kernels = super::engine_kernels();
    let checked = seismic_lang::source::load(&[kernels], seismic_std::sources())
        .expect("the engine kernel library checks")
        .module;
    // Held for the process, as a build embeds a bundle.
    let decoded = decode_checked_bundle(encode_checked_bundle(&checked).leak())
        .expect("the engine kernel bundle decodes");
    let representations = qwen_representations();
    let dialects = [
        (
            BackendName::Metal,
            Dialect::Metal(MetalFeatures { tensor_ops: true }),
        ),
        (BackendName::Cuda, Dialect::Cuda(CudaFeatures::UNOPENED)),
        (BackendName::Vulkan, Dialect::Vulkan(VULKAN)),
    ];
    let mut compared = 0;
    let mut native_entries = 0;
    for info in checked.entries() {
        let decoded_entry = decoded
            .entry_named(&info.name)
            .expect("the decoded module has the entry");
        let natives: Vec<_> = dialects
            .iter()
            .filter_map(|(backend, dialect)| {
                checked
                    .native_implementation(info.id, *backend)
                    .map(|native| (*backend, *dialect, native))
            })
            .collect();
        if natives.is_empty() {
            continue;
        }
        native_entries += 1;
        let admit = |bound: &ElementBindings| info.element_domain.admit(bound).is_ok();
        let parameters = info.element_domain.parameters();
        let mut admissible = bindings(parameters, &representations, admit);
        if admissible.is_empty() {
            // An import from an external representation, outside the Qwen
            // storage set: any registered representation the entry builds for.
            let every: Vec<_> = registry::representations()
                .iter()
                .map(|info| info.id)
                .collect();
            admissible = bindings(parameters, &every, |bound| {
                checked.entry(info.id, bound).is_ok()
            });
        }
        assert!(!admissible.is_empty(), "`{}` admits no binding", info.name);
        for bound in &admissible {
            let from_source = checked
                .entry(info.id, bound)
                .expect("the checked entry builds");
            let from_bundle = decoded
                .entry(decoded_entry, bound)
                .expect("the decoded entry builds");
            for (backend, dialect, native) in &natives {
                let decoded_native = decoded
                    .native_implementation(decoded_entry, *backend)
                    .expect("the decoded module has the native implementation");
                assert_eq!(native.launches, decoded_native.launches);
                // The implementations at the binding, which preparation
                // forms: none where an element read has no value there, in
                // both modules alike.
                let (native, decoded_native) = match (
                    checked.native_implementation_at(info.id, *backend, bound),
                    decoded.native_implementation_at(decoded_entry, *backend, bound),
                ) {
                    (Some(native), Some(decoded_native)) => (native, decoded_native),
                    (None, None) => continue,
                    _ => panic!(
                        "`{}` on {} forms at a binding in one module alone",
                        info.name,
                        backend.as_str()
                    ),
                };
                let specialization = specialization(native);
                fn asset(
                    module: &CheckedModule,
                    entry: seismic_lang::ids::EntryId,
                    backend: BackendName,
                ) -> seismic_lang::checked::NativeAsset<'_> {
                    module.native_asset(entry, backend).expect("the asset is captured")
                }
                let expected = render_source(
                    *dialect,
                    &from_source,
                    bound,
                    native,
                    &specialization,
                    asset(&checked, info.id, *backend),
                );
                let actual = render_source(
                    *dialect,
                    &from_bundle,
                    bound,
                    decoded_native,
                    &specialization,
                    asset(&decoded, decoded_entry, *backend),
                );
                assert!(
                    expected == actual,
                    "`{}` on {} forms a different source from the decoded module",
                    info.name,
                    backend.as_str()
                );
                compared += 1;
            }
        }
    }
    assert!(native_entries > 0 && compared >= native_entries);
    eprintln!("compared {compared} native sources over {native_entries} entries");
}

/// The constant block as rendered before it was rendered in one pass: each
/// module constant the asset names as a whole identifier, found by a scan of
/// the asset for each constant. The oracle [`constant_macros`] must equal.
fn reference_constant_macros(prefix: &mut String, asset: NativeAsset<'_>) {
    for (name, value) in asset.constants {
        let symbol = format!("SEISMIC_CONST_{}", native_macro(name));
        if reference_names_symbol(asset.text, &symbol) {
            prefix.push_str(&format!("#define {symbol} {value}\n"));
        }
    }
}

/// Whether `text` has `symbol` as a whole identifier.
fn reference_names_symbol(text: &str, symbol: &str) -> bool {
    let identifier = |character: char| character.is_ascii_alphanumeric() || character == '_';
    text.match_indices(symbol).any(|(at, _)| {
        !text[..at].chars().next_back().is_some_and(identifier)
            && !text[at + symbol.len()..].chars().next().is_some_and(identifier)
    })
}

/// Configurations rendered per entry instance.
const CONFIGURATIONS: usize = 16;
/// Entries rendered per dialect, of those with the longest assets.
const RENDERED_ENTRIES: usize = 4;

/// Every source of the engine kernel library renders the module constants
/// it names exactly as the per-constant scan did, and a configuration's
/// sources render without a scan of the asset per constant. Prints what
/// rendering takes: the constant block per asset either way, and the
/// sources preparation renders per configuration (each launch's, or the
/// whole entry's), for the entries with the longest assets.
///
/// Slow: renders the whole engine kernel library. Run it in release with
/// `--ignored --nocapture`.
#[test]
#[ignore]
fn kernel_sources_render_their_named_constants_in_one_pass() {
    let kernels = super::engine_kernels();
    let checked = seismic_lang::source::load(&[kernels], seismic_std::sources())
        .expect("the engine kernel library checks")
        .module;
    let dialects = [
        (
            BackendName::Metal,
            Dialect::Metal(MetalFeatures { tensor_ops: true }),
        ),
        (BackendName::Cuda, Dialect::Cuda(CudaFeatures::UNOPENED)),
        (BackendName::Vulkan, Dialect::Vulkan(VULKAN)),
    ];
    // The constant block of every asset, either way, and what each took.
    let mut blocks = BTreeMap::new();
    let (mut naming, mut scanned, mut one_pass) = (0usize, 0f64, 0f64);
    for info in checked.entries() {
        for (backend, _) in &dialects {
            let Some(asset) = checked.native_asset(info.id, *backend) else {
                continue;
            };
            let began = Instant::now();
            let mut expected = String::new();
            reference_constant_macros(&mut expected, asset);
            let reference = began.elapsed().as_secs_f64();
            let began = Instant::now();
            let mut actual = String::new();
            constant_macros(&mut actual, asset);
            let rendered = began.elapsed().as_secs_f64();
            assert!(
                actual == expected,
                "`{}` on {} renders different constants",
                info.name,
                backend.as_str()
            );
            naming += usize::from(!expected.is_empty());
            scanned += reference;
            one_pass += rendered;
            blocks.insert((info.id, *backend), (reference, rendered));
        }
    }
    assert!(naming > 0, "some asset names a module constant");
    eprintln!(
        "{} assets ({naming} naming some of the module's {} constants): constant block {:.3} ms an asset scanned per constant, {:.3} ms in one pass",
        blocks.len(),
        checked.constants().len(),
        scanned / blocks.len() as f64 * 1e3,
        one_pass / blocks.len() as f64 * 1e3,
    );
    // The sources preparation renders per configuration, as it renders
    // them, for each dialect's entries with the longest assets.
    let representations = qwen_representations();
    for (backend, dialect) in &dialects {
        let mut entries: Vec<_> = checked
            .entries()
            .iter()
            .filter_map(|info| {
                checked
                    .native_asset(info.id, *backend)
                    .map(|asset| (asset.text.len(), info))
            })
            .collect();
        entries.sort_by(|left, right| right.0.cmp(&left.0));
        let mut rendered_entries = 0;
        for (length, info) in entries {
            if rendered_entries == RENDERED_ENTRIES {
                break;
            }
            let admit = |bound: &ElementBindings| info.element_domain.admit(bound).is_ok();
            let Some(bound) = bindings(info.element_domain.parameters(), &representations, admit)
                .into_iter()
                .next()
            else {
                continue;
            };
            let logical = checked.entry(info.id, &bound).expect("the checked entry builds");
            let native = checked
                .native_implementation_at(info.id, *backend, &bound)
                .expect("the module has the native implementation");
            let asset = checked.native_asset(info.id, *backend).expect("the asset is captured");
            // An instance whose first binding the static search admits none
            // at is not one of those timed.
            let Some(statics) = statics(native) else {
                continue;
            };
            let mut configurations = Vec::new();
            native
                .walk_admissible(&statics, |specialization| {
                    if let Ok(prepared) = native.prepared(specialization.clone()) {
                        configurations.push(prepared);
                    }
                    configurations.len() < CONFIGURATIONS
                })
                .expect("the declaration walks");
            let began = Instant::now();
            let mut sources = 0usize;
            for specialization in &configurations {
                sources += match dialect {
                    Dialect::Vulkan(_) => {
                        std::hint::black_box(render_source(
                            *dialect,
                            &logical,
                            &bound,
                            native,
                            specialization,
                            asset,
                        ));
                        1
                    }
                    Dialect::Metal(_) | Dialect::Cuda(_) => {
                        let variants = (0..native.launches.len())
                            .map(|ordinal| {
                                vec![super::plan::LaunchVariant {
                                    code: super::plan::code_values(native, specialization, ordinal),
                                    group_size: native.static_group_size(specialization, ordinal),
                                }]
                            })
                            .collect::<Vec<_>>();
                        std::hint::black_box(super::implementation_programs(
                            *dialect,
                            &logical,
                            &bound,
                            native,
                            specialization,
                            asset,
                            &variants,
                        ))
                        .len()
                    }
                };
            }
            rendered_entries += 1;
            let configurations = configurations.len().max(1) as f64;
            let rendered = began.elapsed().as_secs_f64() / configurations;
            let per = sources as f64 / configurations;
            let (reference, one_pass) = blocks[&(info.id, *backend)];
            eprintln!(
                "{} `{}` ({:.0} KB asset): {per:.0} sources a configuration, {:.2} ms a configuration in one pass, {:.2} ms scanned per constant",
                backend.as_str(),
                info.name,
                length as f64 / 1024.,
                rendered * 1e3,
                (rendered + per * (reference - one_pass)) * 1e3,
            );
        }
    }
}
