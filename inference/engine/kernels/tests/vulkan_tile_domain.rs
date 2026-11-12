//! The Vulkan projection GEMM's sub-tile domain, over every kernel the crate
//! declares.

mod common;

use seismic_lang::checked::SourceFile;
use seismic_lang::native_source::identifiers;

/// Every Vulkan implementation whose source includes the projection GEMM
/// template and tunes its sub-tile cannot hold the sub-tile the template
/// states it cannot run, whatever its declaration lists. The implementations
/// are derived from their sources, not listed, and the constraint is asked
/// whether it can hold there, so elements, statics and the rule's spelling do
/// not matter.
#[test]
fn every_vulkan_user_of_the_projection_gemm_excludes_its_spilling_sub_tile() {
    let mut sources = seismic_std::sources();
    // Read at run time, as `common::check` reads the native sources.
    let manifest = std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    let directory = std::path::Path::new(&manifest).join("kernels");
    let mut files = std::fs::read_dir(&directory)
        .expect("the kernel sources are readable")
        .map(|entry| entry.expect("a kernel directory entry").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "seismic"))
        .collect::<Vec<_>>();
    files.sort();
    for path in files {
        sources.push(SourceFile {
            path: path.file_name().unwrap().to_string_lossy().into_owned(),
            text: std::fs::read_to_string(&path).expect("a kernel source is readable"),
        });
    }
    let module = common::check(sources);
    let template = std::fs::read_to_string(directory.join("vulkan/lib/projection/projection.glsl"))
        .expect("the template is readable");
    let stated = |name: &str| -> u64 {
        template
            .lines()
            .find_map(|line| line.strip_prefix(&format!("#define {name} ")))
            .unwrap_or_else(|| panic!("the template states {name}"))
            .trim()
            .parse()
            .expect("the template states an integer")
    };
    let spilling = (
        stated("PROJECTION_GEMM_SPILLING_SUB_M"),
        stated("PROJECTION_GEMM_SPILLING_SUB_N"),
    );
    let mut users = 0;
    for entry in module.entries() {
        let Some(native) = module.native_implementation(entry.id, seismic::BackendName::Vulkan) else {
            continue;
        };
        let asset = module
            .native_asset(entry.id, seismic::BackendName::Vulkan)
            .unwrap_or_else(|| panic!("{} has a Vulkan source", entry.name));
        // The checker's own tokenizer: identifiers outside comments and literals.
        let tokens = identifiers(asset.text);
        let reads = |name: &str| tokens.iter().any(|(token, _)| *token == name);
        let tunes_sub_tile = reads("SEISMIC_TUNE_SUB_M") && reads("SEISMIC_TUNE_SUB_N");
        let includes_template = reads("PROJECTION_GEMM_SPILLING_SUB_M");
        let constraint = native.constraint.as_ref();
        // Whether the constraint can hold with the sub-tile at `at` and every
        // other name unknown.
        let can_hold = |at: (u64, u64)| {
            constraint.map_or(true, |constraint| {
                constraint.may_hold(&|_| None, &|name| match name {
                    "SUB_M" => Some((at.0, at.0)),
                    "SUB_N" => Some((at.1, at.1)),
                    _ => None,
                })
            })
        };
        if tunes_sub_tile && includes_template {
            users += 1;
            assert!(!can_hold(spilling), "{} can hold the spilling sub-tile", entry.name);
            assert!(can_hold((32, 32)), "{} holds no sub-tile at all", entry.name);
        } else {
            assert!(
                can_hold(spilling),
                "{} is constrained though it is not a user of the template",
                entry.name
            );
        }
    }
    assert!(users > 0, "no implementation uses the template");
}
