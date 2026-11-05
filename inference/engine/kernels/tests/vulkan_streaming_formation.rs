// Offline SPIR-V formation of streaming's distinct native source paths.
// No device or model is opened: this checks ABI and shader compilation.
use seismic::coverage::{form_requests, InProcess, Progress, Toolchain};
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

#[test]
fn vulkan_streaming_forms() {
    let module = magnitude_kernels::module().unwrap();
    let toolchain = Toolchain::open(BackendName::Vulkan).expect("Vulkan forms on every host");
    let g = Element::named("gguf_q4_k").unwrap();
    let w = Element::named("q4k@rows16").unwrap();
    let (b, h) = (Element::bf16(), Element::f16());
    let (hd, f) = (2048, 768);
    let mut requests = vec![
        request("repack_weight", &[("E", g), ("U", w)], &[]),
        request("routed_move", &[("U", w)], &[]),
        request("routed_copy", &[("U", w)], &[]),
        request("routed_copy", &[("U", b)], &[]),
        request("routed_list_grid", &[], &[]),
        request("routed_step", &[], &[]),
        request("routed_scratch_wait", &[("SG", g), ("SU", g), ("SD", g)], &[]),
        request("routed_scratch_grids", &[], &[]),
        request("routed_convert", &[("ST", g), ("U", w)], &[]),
        request("routed_convert", &[("ST", b), ("U", h)], &[]),
        request("routed_tier_wait", &[], &[("K", 8)]),
    ];
    // The decode entries over bank and host tier records.
    for weight in [w, b] {
        let decode = [("H", hd), ("K", 8), ("F", f)];
        requests.push(request(
            "routed_gate_up",
            &[("A", b), ("EGW", weight), ("EUW", weight)],
            &decode.iter().copied().chain([("GS", 0), ("US", 0)]).collect::<Vec<_>>(),
        ));
        requests.push(request(
            "routed_up",
            &[("A", b), ("EUW", weight)],
            &decode.iter().copied().chain([("US", 0)]).collect::<Vec<_>>(),
        ));
        requests.push(request("routed_down", &[("A", b), ("R", Element::f32()), ("EDW", weight)], &decode));
        requests.push(request("routed_down_project", &[("A", b), ("EDW", weight)], &decode));
        requests.push(request(
            "routed_output_project",
            &[("A", b), ("EDW", weight), ("SDW", w)],
            &decode.iter().copied().chain([("S", 512)]).collect::<Vec<_>>(),
        ));
        requests.push(request(
            "routed_output",
            &[("A", b), ("EDW", weight), ("SDW", w)],
            &decode.iter().copied().chain([("SN", 1), ("IN", 0), ("S", 512)]).collect::<Vec<_>>(),
        ));
    }
    // The request to the CPU, and the combines that wait for its answer.
    requests.push(request("routed_site_request", &[("A", b)], &[("H", hd), ("K", 8)]));
    requests.push(request("routed_down_sited", &[("A", b), ("R", Element::f32())], &[("H", hd), ("K", 8)]));
    requests.push(request("routed_output_sited", &[("A", b)], &[("H", hd), ("SN", 1), ("IN", 0), ("K", 8)]));
    let route = [
        ("H", 2048),
        ("PT", 1),
        ("NM", 2048),
        ("E", 256),
        ("K", 8),
        ("S", 512),
        ("ES", 0),
    ];
    let route_elements = [("A", b), ("NW", b), ("RW", b), ("SGW", w), ("SUW", w)];
    requests.push(request(
        "routed_route",
        &route_elements[..3],
        &route
            .iter()
            .copied()
            .filter(|(name, _)| *name != "S")
            .collect::<Vec<_>>(),
    ));
    requests.push(request("routed_route_shared", &route_elements, &route));
    let mut route_without_shared = route;
    route_without_shared.iter_mut().find(|(name, _)| *name == "S").unwrap().1 = 0;
    requests.push(request("routed_route_shared", &route_elements, &route_without_shared));
    let select = [("H", 2048), ("E", 64), ("K", 4), ("ES", 0)];
    let select_elements = [("A", b), ("NW", b), ("RNW", b), ("RW", b)];
    requests.push(request("routed_select", &select_elements, &select));
    // Across a link (a tier, TM = E) and on unified memory (none).
    for (ahead, linked) in [(0, true), (2, true), (2, false)] {
        let streaming = |experts: u64| {
            [
                ("AHEAD", ahead),
                ("EN", if ahead == 0 { 1 } else { 32 }),
                ("TM", if linked { experts } else { 0 }),
            ]
        };
        requests.push(request(
            "routed_route_streamed",
            &route_elements,
            &route.iter().copied().chain(streaming(256)).collect::<Vec<_>>(),
        ));
        requests.push(request(
            "routed_select_streamed",
            &select_elements,
            &select.iter().copied().chain(streaming(64)).collect::<Vec<_>>(),
        ));
    }
    // Packed current and next routers bind separate rows16 geometries.
    // NM=0 additionally selects the unnormalized PT-part input recipe.
    let hinted = |experts: u64| [("AHEAD", 2), ("EN", 32), ("TM", experts)];
    let mut packed_route = route_elements;
    packed_route.iter_mut().find(|(name, _)| *name == "RW").unwrap().1 = w;
    let mut packed_select = select_elements;
    packed_select.iter_mut().find(|(name, _)| *name == "RW").unwrap().1 = w;
    requests.push(request(
        "routed_route_streamed",
        &packed_route,
        &route.into_iter().chain(hinted(256)).collect::<Vec<_>>(),
    ));
    requests.push(request(
        "routed_select_streamed",
        &packed_select,
        &select.into_iter().chain(hinted(64)).collect::<Vec<_>>(),
    ));
    let mut unnormalized_route = route;
    unnormalized_route.iter_mut().find(|(name, _)| *name == "NM").unwrap().1 = 0;
    unnormalized_route.iter_mut().find(|(name, _)| *name == "PT").unwrap().1 = 2;
    requests.push(request(
        "routed_route_streamed",
        &route_elements,
        &unnormalized_route.into_iter().chain(hinted(256)).collect::<Vec<_>>(),
    ));
    let results = form_requests(
        module,
        &toolchain,
        &InProcess,
        &requests,
        &Progress::default(),
    );
    for (request, result) in requests.iter().zip(results) {
        result.unwrap_or_else(|error| panic!("{request}: {error:?}"));
    }
}
