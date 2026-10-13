use seismic::coverage::{CoverageError, RequestFormer};
use seismic::{BackendName, Element, KernelRequest};
fn request(
    entry: &'static str,
    elements: &[(&str, Element)],
    statics: &[(&str, u64)],
) -> KernelRequest {
    KernelRequest {
        backend: BackendName::Vulkan,
        entry,
        elements: elements.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        statics: statics.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
    }
}
/// The K8/V4 history plane widths for a `width`-column head, as the engine's
/// affine codec lays them out and `attention_append_k8v4` declares them: 8-bit
/// key and 4-bit value codes packed into u32 words, and a (scale, zero) pair per
/// 32-column group.
fn k8v4_planes(width: u64) -> [(&'static str, u64); 4] {
    let coefficients = 2 * width / 32;
    [
        ("KC", width * 8 / 32),
        ("KP", coefficients),
        ("VC", width * 4 / 32),
        ("VP", coefficients),
    ]
}
#[test]
fn vulkan_priming_abi_forms() {
    let module = magnitude_kernels::module().unwrap();
    let former = match RequestFormer::open(BackendName::Vulkan) {
        Ok(former) => former,
        Err(CoverageError::ToolchainUnavailable(reason))
            if std::env::var_os("MAGNITUDE_REQUIRE_KERNEL_FORMATION").is_none() =>
        {
            eprintln!("skipping focused Vulkan priming ABI formation: {reason}");
            return;
        }
        Err(error) => panic!("Vulkan priming ABI formation did not run: {error}"),
    };
    let mut requests = Vec::new();
    // Qwen 4B fixture geometry; cover all normalization/presence branches.
    let (p, s) = (32, 192);
    for (entry, planes) in [
        ("attention_append_dense", &[][..]),
        ("attention_append_k8v4", &k8v4_planes(2 * p + s)[..]),
    ] {
        for f in [0, 1] {
            for n in [0, 1] {
                for nv in [0, 1] {
                    let statics = [
                        ("KV", 4),
                        ("P", p),
                        ("S", s),
                        ("F", f),
                        ("N", n),
                        ("NV", nv),
                    ];
                    requests.push(request(
                        entry,
                        &[("A", Element::bf16())],
                        &[&statics[..], planes].concat(),
                    ));
                }
            }
        }
    }
    // Full and static-zero query forms use identical K/V projection geometry.
    for q in [0, 8192] {
        for weight in [
            Element::bf16(),
            Element::stored("q8g32s", seismic_lang::registry::Layout::Rows16).unwrap(),
        ] {
            requests.push(request(
                "attention_project",
                &[
                    ("A", Element::bf16()),
                    ("NW", Element::bf16()),
                    ("QW", weight),
                    ("GW", weight),
                    ("KW", weight),
                    ("VW", weight),
                ],
                &[
                    ("D", 2560),
                    ("PT", 1),
                    ("NM", 2560),
                    ("Q", q),
                    ("GR", 0),
                    ("K", 1024),
                    ("V", 1024),
                ],
            ));
        }
    }
    let results = former.form_all(module, &requests);
    for (request, result) in requests.iter().zip(results) {
        eprintln!("{request}: {result:?}");
        assert!(result.is_ok(), "{request}: {result:?}");
    }
    eprintln!("FORMED {} exact native Vulkan requests", requests.len());
}
