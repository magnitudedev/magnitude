//! Every served target's family outline equals its load's layouts.

use super::catalog::{
    LIMITS, SELECTION, bank_reservations, bind_catalog_draft, refused_by_the_device, serves, target_manifest,
};
use crate::{
    ComponentSelection, ExecutionPath, ModelLoadPlan, PlanError, PlannedMethod, ResourceCapacity, ResourcePlanner,
    UnstreamedExperts, resident_layout,
};
use magnitude_artifacts::catalog_headers::ComponentRole;
use magnitude_artifacts::{ArtifactIdentity, ComponentFile, ComponentManifest, PackageIdentity};
use magnitude_family_common::headers;
use magnitude_family_registry::FAMILIES;
use magnitude_state::KvCodec;
use seismic::BackendName;
use std::time::Duration;

/// Every served target's target family, outlined as assessment charges it
/// (`checked_target_family`, from the family's graph traces), holds exactly
/// what the load's layouts of the same plan hold, field by field
/// (`target_outline_differences`): on `backend`, resident and with its
/// experts banked at the floor, plain and with each drafter it loads with
/// (its embedded heads, its catalog draft), each as drafted and undrafted
/// (taps and conditioning residual written, or not). The mean time of an
/// outline and of the layouts is printed.
fn every_served_target_outlines_its_family_as_its_load_lays_it_out(backend: BackendName) {
    let identity = PackageIdentity { target: ArtifactIdentity([7; 32]), projector: None };
    let served = serves();
    let (mut rows, mut failures) = (0u32, Vec::new());
    let (mut time, mut graphs) = ([Duration::ZERO; 2], 0usize);
    let index = headers::index();
    for component in index.iter().filter(|component| component.role == ComponentRole::Target) {
        let name = component.model.as_str();
        if !served(name) {
            continue;
        }
        let target = headers::component(component);
        let family = FAMILIES
            .iter()
            .find(|family| family.recognizes(&target))
            .unwrap_or_else(|| panic!("{name}: a family recognizes the served target"));
        let declared = family
            .inspect(&target, None, identity)
            .unwrap_or_else(|error| panic!("{name}: its family inspects it: {error}"));
        if crate::operators::admit(&declared, false).is_err() {
            continue;
        }
        // Plain, with its embedded heads, and with its catalog draft.
        let mut loads = vec![("plain", SELECTION, declared.clone(), None)];
        if declared.head.is_some() {
            loads.push(("mtp", ComponentSelection { head: true, vision: false }, declared.clone(), None));
        }
        let draft = index
            .iter()
            .find(|draft| draft.model == component.model && draft.role == ComponentRole::Draft);
        if let Some(draft) = draft {
            let directory = headers::component(draft);
            let definition = bind_catalog_draft(*family, &declared, &directory)
                .unwrap_or_else(|error| panic!("{name}: its catalog draft binds: {error}"));
            let manifest = ComponentManifest {
                files: vec![ComponentFile {
                    path: draft.path.clone(),
                    size: directory.tensors.iter().map(|tensor| tensor.nbytes).sum(),
                }],
                identity: ArtifactIdentity([9; 32]),
                tensors: directory.tensors.clone(),
            };
            loads.push(("drafter", ComponentSelection { head: true, vision: false }, definition, Some(manifest)));
        }
        for (method, selection, definition, draft) in loads {
            if crate::operators::admit(&definition, selection.head).is_err() {
                continue;
            }
            let manifest = magnitude_artifacts::PackageManifest {
                draft,
                ..target_manifest(component, &target, identity)
            };
            let layout = resident_layout(ExecutionPath::Native, backend);
            // A load need not have a resident form on every backend.
            let Ok(load) =
                ModelLoadPlan::derive(&manifest, &definition, selection, layout, PlannedMethod::Plain, LIMITS)
            else {
                continue;
            };
            // Experts are banked where they stream: a model that routes
            // none, or a device that streams none, has no banked form.
            let banked = match bank_reservations(&definition, backend) {
                Ok((reservation, host)) => Some(load.bank_floor(&definition, 1, LIMITS.flights(), reservation, host).and_then(|floor| {
                    load.clone().with_expert_bank(&definition, floor, 1, LIMITS.flights(), reservation, host)
                })),
                Err(PlanError::Unstreamed(reason))
                    if matches!(reason, UnstreamedExperts::Unrouted) || refused_by_the_device(&reason) =>
                {
                    None
                }
                Err(error) => Some(Err(error)),
            };
            let forms = std::iter::once(("resident", Ok(load))).chain(banked.map(|banked| ("banked", banked)));
            for (form, load) in forms {
                let line = format!("{name} {method} {} {form}", backend.as_str());
                let program = match load.and_then(|load| {
                    let program = load.program_plan(&definition, KvCodec::AffineK8V4)?;
                    Ok((load, program))
                }) {
                    Ok(planned) => planned,
                    Err(error) => {
                        failures.push(format!("{line}: it plans: {error}"));
                        continue;
                    }
                };
                let (load, program) = program;
                let state = ResourcePlanner::state_plan(
                    &definition,
                    &load,
                    PlannedMethod::Plain,
                    KvCodec::AffineK8V4,
                    LIMITS,
                    ResourceCapacity { domain_bytes: 64 << 30, tensor_operations: crate::TensorOperations::Absent },
                )
                .unwrap_or_else(|error| panic!("{line}: its state plans: {error}"));
                let (differences, [outlining, laying], outlined) =
                    crate::programs::native_target_graph::target_outline_differences(
                        backend,
                        &load,
                        &definition.decoder,
                        &state,
                        program.target(),
                        LIMITS,
                        load.pass(LIMITS),
                    );
                rows += 1;
                time = [time[0] + outlining, time[1] + laying];
                graphs += outlined;
                failures.extend(differences.into_iter().map(|difference| format!("{line}: {difference}")));
            }
        }
    }
    let mean = |total: Duration| total / rows.max(1);
    println!(
        "{rows} rows (each a drafted and an undrafted family); an outline {:?}, the layouts {:?}, on mean; \
         {graphs} graphs outlined, {} traces recorded",
        mean(time[0]),
        mean(time[1]),
        crate::programs::native_target_graph::recorded_traces(),
    );
    assert!(rows > 0);
    assert!(failures.is_empty(), "{} differences:\n{}", failures.len(), failures.join("\n"));
}

#[test]
#[ignore = "the served catalog's headers: run on a rig host"]
fn every_served_target_outlines_its_family_as_its_load_lays_it_out_on_cpu() {
    every_served_target_outlines_its_family_as_its_load_lays_it_out(BackendName::Cpu);
}

#[test]
#[ignore = "the served catalog's headers: run on a rig host"]
fn every_served_target_outlines_its_family_as_its_load_lays_it_out_on_metal() {
    every_served_target_outlines_its_family_as_its_load_lays_it_out(BackendName::Metal);
}

#[test]
#[ignore = "the served catalog's headers: run on a rig host"]
fn every_served_target_outlines_its_family_as_its_load_lays_it_out_on_cuda() {
    every_served_target_outlines_its_family_as_its_load_lays_it_out(BackendName::Cuda);
}

#[test]
#[ignore = "the served catalog's headers: run on a rig host"]
fn every_served_target_outlines_its_family_as_its_load_lays_it_out_on_vulkan() {
    every_served_target_outlines_its_family_as_its_load_lays_it_out(BackendName::Vulkan);
}
