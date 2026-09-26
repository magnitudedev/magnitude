//! Every Vulkan native implementation of the engine kernel library forms
//! (glslang, seal, SPIRV-Tools validation) under every device configuration
//! the Vulkan floor admits, without a device.

use super::abi::{render_source, vulkan::VulkanFeatures, Dialect};
use seismic_lang::checked::{ElementParameter, NativeImplementation, NativeSpecialization};
use seismic_lang::entry::ElementBindings;
use seismic_lang::ids::RepresentationId;
use seismic_lang::registry::{self, BackendName, Layout};
use std::path::PathBuf;

/// Bindings formed per entry, at most.
const BINDINGS_PER_ENTRY: usize = 4;

/// A deterministic admissible specialization of `native`.
fn specialization(native: &NativeImplementation) -> NativeSpecialization {
    const VALUES: [u64; 14] = [
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 2560, 4096,
    ];
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..20_000 {
        let statics = native
            .statics
            .iter()
            .fold(NativeSpecialization::new(), |statics, name| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                statics.with_static(name.as_str(), VALUES[(state >> 33) as usize % VALUES.len()])
            });
        if let Ok(specialization) = native.default_specialization(&statics) {
            return specialization;
        }
    }
    panic!(
        "no admissible specialization of `{}` found",
        native.source_path
    )
}

/// Candidate bindings collected per entry, at most.
const CANDIDATES_PER_ENTRY: usize = 16;

/// The first [`CANDIDATES_PER_ENTRY`] bindings of `parameters` over
/// `representations` that `admit` accepts, in a fixed order.
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
            if found.len() == CANDIDATES_PER_ENTRY {
                return found;
            }
        }
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

/// Slow: forms the Vulkan launches of the engine kernel library under every
/// configuration. Run it in release with `--ignored`.
///
/// A binding whose asset rejects it with `#error` under the reference
/// configuration is not one the asset implements; every other binding must
/// form under every configuration, and every entry must form at least one.
#[test]
#[ignore]
fn every_vulkan_kernel_forms_under_every_floor_configuration() {
    let kernels =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../engine/model-kernels/kernels");
    let checked = seismic_lang::source::load(&[kernels], seismic_std::sources())
        .expect("the engine kernel library checks")
        .module;
    let mut storage: Vec<_> = ["bf16", "f16", "f32"]
        .into_iter()
        .map(|name| registry::representation(name).expect("dense representation"))
        .collect();
    for packed in ["q4k", "q6k", "q8g32s"] {
        storage.push(registry::storage(packed, Layout::Rows16).expect("rows16 storage"));
    }
    let every: Vec<_> = registry::representations()
        .iter()
        .map(|info| info.id)
        .collect();
    let configurations = configurations();
    let entries: Vec<_> = checked
        .entries()
        .iter()
        .filter(|info| {
            checked
                .native_implementation(info.id, BackendName::Vulkan)
                .is_some()
        })
        .collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let formed = std::sync::atomic::AtomicUsize::new(0);
    let failures = std::sync::Mutex::new(Vec::new());
    let workers = std::thread::available_parallelism().map_or(1, usize::from);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(info) = entries.get(index) else {
                    return;
                };
                let (count, entry_failures) =
                    form_entry(&checked, info, &storage, &every, &configurations);
                formed.fetch_add(count, std::sync::atomic::Ordering::Relaxed);
                failures
                    .lock()
                    .expect("no worker panics holding the lock")
                    .extend(entry_failures);
            });
        }
    });
    let failures = failures.into_inner().expect("the lock is not poisoned");
    eprintln!(
        "formed {} Vulkan launches of {} entries",
        formed.into_inner(),
        entries.len()
    );
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Forms up to [`BINDINGS_PER_ENTRY`] implemented bindings of `info` under
/// every configuration: the launches formed and the failures.
fn form_entry(
    checked: &seismic_lang::checked::CheckedModule,
    info: &seismic_lang::checked::EntryInfo,
    storage: &[RepresentationId],
    every: &[RepresentationId],
    configurations: &[Configuration],
) -> (usize, Vec<String>) {
    let native = checked
        .native_implementation(info.id, BackendName::Vulkan)
        .expect("the entry has a Vulkan implementation");
    let parameters = info.element_domain.parameters();
    let mut candidates = bindings(parameters, storage, |bound| {
        info.element_domain.admit(bound).is_ok()
    });
    if candidates.is_empty() {
        // An import from an external representation.
        candidates = bindings(parameters, every, |bound| {
            checked.entry(info.id, bound).is_ok()
        });
    }
    let asset = checked
        .native_asset(info.id, BackendName::Vulkan)
        .expect("the asset is captured");
    let specialization = specialization(native);
    let mut formed = 0;
    let mut failures = Vec::new();
    let mut implemented = 0;
    for bound in &candidates {
        if implemented == BINDINGS_PER_ENTRY {
            break;
        }
        let logical = checked.entry(info.id, bound).expect("the entry builds");
        let form = |(_, features, environment): &Configuration| {
            let source = render_source(
                Dialect::Vulkan(*features),
                &logical,
                bound,
                native,
                &specialization,
                asset,
            );
            native
                .launches
                .iter()
                .map(|launch| {
                    (
                        &launch.kernel,
                        seismic_vulkan::formation::compile(&source, &launch.kernel, *environment),
                    )
                })
                .collect::<Vec<_>>()
        };
        let reference = form(&configurations[0]);
        let rejected = reference.iter().any(|(_, result)| {
            matches!(
                result,
                Err(seismic_native_target::NativeCompilationError::ToolchainFailure(log))
                    if log.contains("'#error'")
            )
        });
        if rejected {
            continue;
        }
        implemented += 1;
        for (ordinal, configuration) in configurations.iter().enumerate() {
            let results = if ordinal == 0 {
                reference.clone()
            } else {
                form(configuration)
            };
            for (kernel, result) in results {
                let failure = match result {
                    Ok(module) => check_module(&module, &configuration.1).err(),
                    Err(error) => Some(format!("{error:?}")),
                };
                match failure {
                    Some(reason) => failures.push(format!(
                        "`{}` {kernel} {bound:?} [{}]: {reason}",
                        info.name, configuration.0
                    )),
                    None => formed += 1,
                }
            }
        }
    }
    if implemented == 0 {
        failures.push(format!("`{}` implements no candidate binding", info.name));
    }
    (formed, failures)
}

type Configuration = (
    &'static str,
    VulkanFeatures,
    seismic_vulkan::seal::Environment,
);

/// The configurations the Vulkan floor admits: 32- and 64-lane hardware
/// subgroups, with and without fp16 arithmetic, cooperative matrix where
/// 32-lane fp16 hardware has it, and the NVIDIA environment (fp32 RTE
/// probed, not declared).
fn configurations() -> Vec<Configuration> {
    let declared = seismic_vulkan::seal::Environment {
        float16: true,
        rounding_rte_32: true,
        denorm_preserve_32: true,
    };
    let fp32 = seismic_vulkan::seal::Environment {
        float16: false,
        ..declared
    };
    let base = VulkanFeatures {
        subgroup_lanes: 32,
        float16: true,
        matrix: false,
        wide_accumulators: true,
        mixed_dot: true,
        f32_atomic_add: false,
        shared_int64_atomics: false,
    };
    let without_float16 = VulkanFeatures {
        float16: false,
        ..base
    };
    vec![
        ("32 lanes, fp16", base, declared),
        (
            "32 lanes, fp16, matrix",
            VulkanFeatures {
                matrix: true,
                ..base
            },
            declared,
        ),
        (
            "32 lanes, fp32 only, rte32 probed",
            without_float16,
            seismic_vulkan::seal::Environment {
                rounding_rte_32: false,
                denorm_preserve_32: false,
                ..fp32
            },
        ),
        (
            "64 lanes, fp16",
            VulkanFeatures {
                subgroup_lanes: 64,
                ..base
            },
            declared,
        ),
        (
            "64 lanes, fp32 only",
            VulkanFeatures {
                subgroup_lanes: 64,
                ..without_float16
            },
            fp32,
        ),
    ]
}

/// `OpCapability` and the `Float16` capability.
const OP_CAPABILITY: u32 = 17;
const CAPABILITY_FLOAT16: u32 = 9;

/// A module for a device without fp16 arithmetic declares no `Float16`
/// capability.
fn check_module(module: &[u32], features: &VulkanFeatures) -> Result<(), String> {
    if features.float16 {
        return Ok(());
    }
    let mut at = 5;
    while at < module.len() {
        let words = (module[at] >> 16) as usize;
        if module[at] & 0xffff == OP_CAPABILITY && module[at + 1] == CAPABILITY_FLOAT16 {
            return Err("declares the Float16 capability without fp16 arithmetic".into());
        }
        at += words.max(1);
    }
    Ok(())
}
