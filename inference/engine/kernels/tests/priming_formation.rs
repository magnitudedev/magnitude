use seismic::coverage::{form_requests, InProcess, Progress, Toolchain};
use seismic::{BackendName, Element, Entry, KernelRequest};
/// The Vulkan request of `E` at `elements`, binding every static its own
/// declaration names from `form`: a declared static the form has no value
/// for fails by name, so the requests follow the declarations as they grow.
fn request<E: Entry>(elements: &[(&str, Element)], form: &[(&str, u64)]) -> KernelRequest {
    let implementation =
        seismic::generated::native_implementation_for_backend::<E>(BackendName::Vulkan)
            .unwrap()
            .unwrap_or_else(|| panic!("{} has a Vulkan implementation", E::NAME));
    let statics = implementation
        .statics
        .iter()
        .map(|name| {
            let (_, value) = form
                .iter()
                .find(|(candidate, _)| candidate == name)
                .unwrap_or_else(|| {
                    panic!("{} declares `{name}` static, which the form does not bind", E::NAME)
                });
            (name.clone(), *value)
        })
        .collect();
    KernelRequest {
        backend: BackendName::Vulkan,
        entry: E::NAME,
        elements: elements.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        statics,
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
    // glslang is linked: Vulkan forms on every host.
    let toolchain = Toolchain::open(BackendName::Vulkan).expect("Vulkan forms on every host");
    let mut requests = Vec::new();
    // Qwen 4B fixture geometry; cover all normalization/presence branches.
    let (p, s) = (32, 192);
    for planes in [&[][..], &k8v4_planes(2 * p + s)[..]] {
        for f in [0, 1] {
            for n in [0, 1] {
                for nv in [0, 1] {
                    // Rotary halves split, own values, keys unrounded.
                    let form = [
                        &[
                            ("KV", 4),
                            ("P", p),
                            ("S", s),
                            ("F", f),
                            ("N", n),
                            ("NV", nv),
                            ("RA", 0),
                            ("RT", 0),
                            ("VK", 0),
                            ("KR", 0),
                        ][..],
                        planes,
                    ]
                    .concat();
                    let elements = [("A", Element::bf16())];
                    requests.push(if planes.is_empty() {
                        request::<magnitude_kernels::attention_append_dense::Entry>(&elements, &form)
                    } else {
                        request::<magnitude_kernels::attention_append_k8v4::Entry>(&elements, &form)
                    });
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
            requests.push(request::<magnitude_kernels::attention_project::Entry>(
                &[
                    ("A", Element::bf16()),
                    ("Y", Element::bf16()),
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
                    ("QS", 0),
                    ("GS", 0),
                    ("KS", 0),
                    ("VS", 0),
                ],
            ));
        }
    }
    let results = form_requests(
        module,
        &toolchain,
        &InProcess,
        &requests,
        &Progress::default(),
    );
    for (request, result) in requests.iter().zip(results) {
        eprintln!("{request}: {result:?}");
        assert!(result.is_ok(), "{request}: {result:?}");
    }
    eprintln!("FORMED {} exact native Vulkan requests", requests.len());
}
