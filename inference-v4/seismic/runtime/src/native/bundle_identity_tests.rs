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

use super::abi::{render_source, vulkan::VulkanFeatures, Dialect};
use seismic_lang::bundle::{decode_checked_bundle, encode_checked_bundle};
use seismic_lang::checked::{CheckedModule, ElementParameter, NativeImplementation, NativeSpecialization};
use seismic_lang::entry::ElementBindings;
use seismic_lang::ids::RepresentationId;
use seismic_lang::registry::{self, BackendName, Layout};
use std::path::PathBuf;

const VULKAN: VulkanFeatures = VulkanFeatures {
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
    const VALUES: [u64; 14] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 2560, 4096];
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..20_000 {
        let statics = native.statics.iter().fold(NativeSpecialization::new(), |statics, name| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            statics.with_static(name.as_str(), VALUES[(state >> 33) as usize % VALUES.len()])
        });
        if let Ok(specialization) = native.default_specialization(&statics) {
            return specialization;
        }
    }
    panic!("no admissible specialization of `{}` found", native.source_path)
}

/// Slow: checks the whole engine kernel library. Run it in release with
/// `--ignored`.
#[test]
#[ignore]
fn decoded_kernel_bundle_forms_identical_native_sources() {
    let kernels = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../engine/model-kernels/kernels");
    let checked = seismic_lang::source::load(&[kernels], seismic_std::sources())
        .expect("the engine kernel library checks")
        .module;
    let decoded = decode_checked_bundle(&encode_checked_bundle(&checked))
        .expect("the engine kernel bundle decodes");
    let representations = qwen_representations();
    let dialects = [
        (BackendName::Metal, Dialect::Metal),
        (BackendName::Cuda, Dialect::Cuda),
        (BackendName::Vulkan, Dialect::Vulkan(VULKAN)),
    ];
    let mut compared = 0;
    let mut native_entries = 0;
    for info in checked.entries() {
        let decoded_entry = decoded.entry_named(&info.name).expect("the decoded module has the entry");
        let natives: Vec<_> = dialects
            .iter()
            .filter_map(|(backend, dialect)| {
                checked.native_implementation(info.id, *backend).map(|native| (*backend, *dialect, native))
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
            let every: Vec<_> = registry::representations().iter().map(|info| info.id).collect();
            admissible = bindings(parameters, &every, |bound| checked.entry(info.id, bound).is_ok());
        }
        assert!(!admissible.is_empty(), "`{}` admits no binding", info.name);
        for bound in &admissible {
            let from_source = checked.entry(info.id, bound).expect("the checked entry builds");
            let from_bundle = decoded.entry(decoded_entry, bound).expect("the decoded entry builds");
            for (backend, dialect, native) in &natives {
                let decoded_native = decoded
                    .native_implementation(decoded_entry, *backend)
                    .expect("the decoded module has the native implementation");
                assert_eq!(native.launches, decoded_native.launches);
                let specialization = specialization(native);
                let asset = |module: &CheckedModule, entry| {
                    module.native_asset(entry, *backend).expect("the asset is captured").to_owned()
                };
                let expected = render_source(
                    *dialect,
                    &from_source,
                    bound,
                    native,
                    &specialization,
                    &asset(&checked, info.id),
                );
                let actual = render_source(
                    *dialect,
                    &from_bundle,
                    bound,
                    decoded_native,
                    &specialization,
                    &asset(&decoded, decoded_entry),
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
