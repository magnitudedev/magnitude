//! Every served catalog target with a projector outlines its vision graphs
//! from the family's graph traces (`seismic::trace`: recorded once per
//! topology, answering every later model its guards and shapes admit)
//! exactly as its load lays them out, class by class, on every backend at
//! the standard service's limits and the backend's image cells: storage,
//! census, both host heaps and the bytes of each constant its binding
//! uploads. How many traces were recorded for how many
//! outlines, and the time of an outline a trace answered against one it
//! recorded, are printed.

use super::catalog::{served_targets, LIMITS};
use crate::programs::native_constants::GraphConstant;
use crate::programs::native_vision::{
    certified_vision_family, image_patch_classes, vision_traces_recorded,
};
use crate::{
    image_cell_limit, max_image_cells, resident_layout, ComponentSelection, ExecutionPath,
    ModelLoadPlan, PlannedMethod, ResourceLimits,
};
use magnitude_artifacts::catalog_headers::ComponentRole;
use magnitude_artifacts::{ArtifactIdentity, ComponentFile, ComponentManifest, PackageIdentity, PackageManifest};
use magnitude_family_common::headers;
use magnitude_family_registry::FAMILIES;
use seismic::{BackendName, NativeGraphLayout, NativeGraphOutline};
use std::time::{Duration, Instant};

fn every_served_projector_outlines_its_vision_graphs_from_traces(backend: BackendName) {
    let identity = PackageIdentity {
        target: ArtifactIdentity([7; 32]),
        projector: Some(ArtifactIdentity([8; 32])),
    };
    let (mut rows, mut failures) = (0, Vec::new());
    let (mut answered, mut recorded) = ((0u32, Duration::ZERO), (0u32, Duration::ZERO));
    for (name, _, manifest) in served_targets() {
        let component = |role| {
            headers::index()
                .iter()
                .find(|component| component.model == name && component.role == role)
        };
        let Some(projector_component) = component(ComponentRole::Projector) else {
            continue;
        };
        let target = headers::component(component(ComponentRole::Target).expect("a served target"));
        let projector = headers::component(projector_component);
        let definition = FAMILIES
            .iter()
            .find(|family| family.recognizes(&target))
            .expect("a family recognizes the served target")
            .inspect(&target, Some(&projector), identity)
            .unwrap_or_else(|error| panic!("{name}: its family inspects it with its projector: {error}"));
        let description = definition.vision.as_ref().expect("a projector's definition has vision");
        let manifest = PackageManifest {
            identity,
            projector: Some(ComponentManifest {
                files: vec![ComponentFile {
                    path: projector_component.path.clone(),
                    size: projector.tensors.iter().map(|tensor| tensor.nbytes).sum(),
                }],
                identity: identity.projector.expect("a projector identity"),
                tensors: projector.tensors.clone(),
            }),
            ..manifest
        };
        let cells = image_cell_limit(&definition, LIMITS.max_launch_rows, max_image_cells(backend))
            .unwrap_or_else(|error| panic!("{name}: its image cells: {error}"));
        let limits = ResourceLimits { max_image_cells: cells, ..LIMITS };
        // A target need not have a resident form on every backend.
        let Ok(load) = ModelLoadPlan::derive(
            &manifest,
            &definition,
            ComponentSelection { head: false, vision: true },
            resident_layout(ExecutionPath::Native, backend),
            PlannedMethod::Plain,
            limits,
        ) else {
            continue;
        };
        let classes = image_patch_classes(cells, description)
            .unwrap_or_else(|error| panic!("{name}: its patch classes: {error}"));
        let traces = vision_traces_recorded();
        let began = Instant::now();
        let outlined = certified_vision_family::<NativeGraphOutline>(backend, &load, description, &classes);
        let elapsed = began.elapsed();
        let tally = match vision_traces_recorded() == traces {
            true => &mut answered,
            false => &mut recorded,
        };
        *tally = (tally.0 + 1, tally.1 + elapsed);
        let certified = certified_vision_family::<NativeGraphLayout>(backend, &load, description, &classes);
        rows += 1;
        let line = format!("{name} {} ({} classes)", backend.as_str(), classes.len());
        match (outlined, certified) {
            (Ok((storage, outlines)), Ok((held, layouts))) => {
                let constant_bytes = |constants: &[GraphConstant]| {
                    constants.iter().map(GraphConstant::storage_bytes).collect::<Result<Vec<_>, _>>()
                };
                let outlined = outlines
                    .iter()
                    .map(|(outline, constants)| {
                        (outline.storage_bytes(), outline.plan_heap(), outline.bound_heap(), constant_bytes(constants))
                    })
                    .collect::<Vec<_>>();
                let laid = layouts
                    .iter()
                    .map(|(layout, constants)| {
                        (layout.storage_bytes(), layout.plan_heap(), layout.bound_heap(), constant_bytes(constants))
                    })
                    .collect::<Vec<_>>();
                if storage != held || outlined != laid {
                    failures.push(format!("{line}: outlined {storage:?} {outlined:?}, laid out {held:?} {laid:?}"));
                }
            }
            (outlined, certified) => failures.push(format!(
                "{line}: outlined {:?}, laid out {:?}",
                outlined.err(),
                certified.err()
            )),
        }
    }
    let mean = |(count, total): (u32, Duration)| total.checked_div(count.max(1)).unwrap_or_default();
    println!(
        "{rows} rows; {} traces recorded for {} outlines ({:?} each), {} outlines answered by a trace ({:?} each)",
        vision_traces_recorded(),
        recorded.0,
        mean(recorded),
        answered.0,
        mean(answered)
    );
    assert!(rows > 0);
    assert!(failures.is_empty(), "{} of {rows} rows differ:\n{}", failures.len(), failures.join("\n"));
}

#[test]
#[ignore = "the served catalog's headers: run on a rig host"]
fn every_served_projector_outlines_its_vision_graphs_from_traces_on_cpu() {
    every_served_projector_outlines_its_vision_graphs_from_traces(BackendName::Cpu);
}

#[test]
#[ignore = "the served catalog's headers: run on a rig host"]
fn every_served_projector_outlines_its_vision_graphs_from_traces_on_metal() {
    every_served_projector_outlines_its_vision_graphs_from_traces(BackendName::Metal);
}

#[test]
#[ignore = "the served catalog's headers: run on a rig host"]
fn every_served_projector_outlines_its_vision_graphs_from_traces_on_cuda() {
    every_served_projector_outlines_its_vision_graphs_from_traces(BackendName::Cuda);
}

#[test]
#[ignore = "the served catalog's headers: run on a rig host"]
fn every_served_projector_outlines_its_vision_graphs_from_traces_on_vulkan() {
    every_served_projector_outlines_its_vision_graphs_from_traces(BackendName::Vulkan);
}
