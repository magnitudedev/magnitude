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
                    ("IY", Element::bf16()),
                    ("NW", Element::bf16()),
                    ("QW", weight),
                    ("GW", weight),
                    ("KW", weight),
                    ("VW", weight),
                    ("IQW", weight),
                    ("IKW", weight),
                ],
                &[
                    ("D", 2560),
                    ("PT", 1),
                    ("NM", 2560),
                    ("Q", q),
                    ("GR", 0),
                    ("K", 1024),
                    ("V", 1024),
                    ("IQ", 0),
                    ("IK", 0),
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
    let failed = requests
        .iter()
        .zip(results)
        .filter(|(request, result)| {
            eprintln!("{request}: {result:?}");
            result.is_err()
        })
        .map(|(request, _)| request.to_string())
        .collect::<Vec<_>>();
    assert!(failed.is_empty(), "{} of {} requests fail to form: {failed:#?}", failed.len(), requests.len());
    eprintln!("FORMED {} exact native Vulkan requests", requests.len());
}

/// Every weight whose Vulkan decoder stages a table in shared memory
/// (`form(W, TABLE)`, `packets_stage_tables`) forms in a single projection
/// (`dense_output`) and in a segmented one (`attention_project`) whose four
/// slots read two tables, one twice (staged once) and one of each other kind,
/// so every staged reader, copy and place compiles (and, where the slots'
/// tables pass the bound, every read in place), and the module seals without
/// 64-bit division (`seal.rs`).
#[test]
fn vulkan_staged_table_projections_form() {
    use seismic_lang::forms::Form;
    use seismic_lang::registry::{self, Layout, RepresentationKind};
    let module = magnitude_kernels::module().unwrap();
    let toolchain = Toolchain::open(BackendName::Vulkan).expect("Vulkan forms on every host");
    let table = Form::named(BackendName::Vulkan, "TABLE").expect("Vulkan stages tables");
    let staged: Vec<_> = registry::representations()
        .iter()
        .filter(|info| matches!(&info.kind, RepresentationKind::PackedRows(rows) if rows.layout == Layout::Rows16))
        .filter(|info| table.of(info.id) > 0)
        .collect();
    let plain = Element::stored("q8g32s", Layout::Rows16).unwrap();
    // A segmented projection's weights take one operand recipe, the plain
    // weight's (rounded rows).
    let segmented: Vec<Element> = staged
        .iter()
        .filter(|info| info.operands == registry::OperandRecipe::Rounded)
        .map(|info| Element::named(info.name).expect("a registered representation"))
        .collect();
    assert!(segmented.len() >= 2, "some rows16 weights of the plain recipe stage tables");
    let mut requests = Vec::new();
    for info in &staged {
        requests.push(request::<magnitude_kernels::dense_output::Entry>(
            &[("A", Element::bf16()), ("DW", Element::named(info.name).expect("a registered representation"))],
            &[("H", 2048), ("F", 4096), ("DS", 0)],
        ));
    }
    for (index, &weight) in segmented.iter().enumerate() {
        let other = segmented[(index + 1) % segmented.len()];
        requests.push(request::<magnitude_kernels::attention_project::Entry>(
            &[
                ("A", Element::bf16()),
                ("Y", Element::bf16()),
                ("IY", Element::f32()),
                ("NW", Element::bf16()),
                ("QW", weight),
                ("GW", plain),
                ("KW", other),
                ("VW", weight),
                ("IQW", Element::bf16()),
                ("IKW", Element::bf16()),
            ],
            &[
                ("D", 2048),
                ("PT", 1),
                ("NM", 2048),
                ("Q", 4096),
                ("GR", 4096),
                ("K", 512),
                ("V", 512),
                ("IQ", 2048),
                ("IK", 128),
                ("QS", 0),
                ("GS", 0),
                ("KS", 0),
                ("VS", 0),
            ],
        ));
    }
    let results = form_requests(module, &toolchain, &InProcess, &requests, &Progress::default());
    let failed = requests
        .iter()
        .zip(results)
        .filter_map(|(request, result)| result.err().map(|error| format!("{request}: {error}")))
        .collect::<Vec<_>>();
    assert!(failed.is_empty(), "{} of {} requests fail to form: {failed:#?}", failed.len(), requests.len());
}
