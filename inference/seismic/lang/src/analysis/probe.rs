/// Prints the analysis of the kernel entries named in `PROBE` (all when
/// unset), with each entry's analysis wall time.
#[test]
#[ignore]
fn probe_kernels() {
    use crate::entry::ElementBindings;
    let module = super::tests::kernels();
    let filter = std::env::var("PROBE").unwrap_or_default();
    for info in module.entries() {
        if !filter.is_empty() && !filter.split(',').any(|name| info.name == name) {
            continue;
        }
        let mut bindings = ElementBindings::new();
        for parameter in info.element_domain.parameters() {
            let preferred = if parameter.name.ends_with('W') { ["q4k", "q8g32s", "f32", "bf16"] } else { ["f32", "bf16", "f16", "q4k"] };
            let chosen = preferred.iter().filter_map(|name| crate::registry::representation(name)).find(|id| parameter.uses.admits(*id));
            if let Some(id) = chosen { bindings = bindings.bind(&parameter.name, id); }
        }
        // The representations chosen above are a guess; an entry they do
        // not build is skipped.
        let built = std::panic::catch_unwind(|| module.entry(info.id, &bindings));
        let Ok(Ok(entry)) = built else {
            println!("== {}: not built with {:?}", info.name, bindings.iter().collect::<Vec<_>>());
            continue;
        };
        let started = std::time::Instant::now();
        let analysis = super::analyze(&entry);
        println!("== {} ({:?})", info.name, started.elapsed());
        for parameter in &analysis.parameters {
            println!("  stored {}: {:?}", parameter.name, parameter.stored);
        }
        for grid in &analysis.grids {
            let coordinates = grid.coordinates.iter().map(|c| format!("i{} in {}..{}", c.binder, c.start, c.end)).collect::<Vec<_>>();
            println!("  grid {} {:?} at {:?}: code {} levels {:?} slope {} over [{}]", grid.parameter, grid.plane, grid.index, grid.code, grid.levels, grid.slope, coordinates.join(", "));
        }
        for reduction in &analysis.reductions {
            let text = reduction.extent.to_string();
            println!("  red {:?} over {} coords: {}", reduction.accumulator, reduction.coordinates.len(), &text[..text.len().min(200)]);
        }
    }
}
