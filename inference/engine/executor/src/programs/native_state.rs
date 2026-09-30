//! Native state maintenance over validated row maps and pinned state planes.

use super::{ReadySubmission, StateProgram};
use crate::{
    native::AttestedState,
    programs::graph::{draft::GraphDraft, GraphError},
    DeviceError, InvariantError, NativeGraphWorkspaceLease, StateLaunchCore, StateStorePlan,
    StateWork, SubmitError, ValidatedStateLaunch,
};
use magnitude_kernels::copy_rows;
use seismic::{
    BackendName, Device, Element, NativeGraphBindings, NativeGraphClassSlice, NativeGraphFamily,
    NativeGraphFamilySlot, NativeGraphLayout, NativeGraphMetadata, NativeGraphOutputs,
    NativeGraphPlan, NativeGraphStorageBytes, NativePort, Tensor,
};
use std::rc::Rc;

fn invalid(detail: impl Into<String>) -> SubmitError {
    SubmitError::Invariant(InvariantError {
        context: "native state program",
        detail: detail.into(),
    })
}
fn device(error: impl ToString) -> SubmitError {
    SubmitError::Device(DeviceError::Execution(error.to_string()))
}
pub struct NativeStateProgram {
    graphs: Rc<PreparedStateCopyGraphs>,
}

/// The exact state-copy classes prepared for the target and optional head
/// state planes. Assessment uses this list before opening a device.
pub(crate) fn state_copy_classes(
    target: &StateStorePlan,
    head: Option<&StateStorePlan>,
    row_classes: &[u64],
) -> Result<Vec<StateCopyGraphClass>, String> {
    let mut classes = Vec::new();
    for state in std::iter::once(target).chain(head) {
        // Every stored history domain's planes, over its own rows and slabs.
        let planes = state.domains.iter().flat_map(|domain| {
            domain
                .components
                .iter()
                .flat_map(|component| component.planes())
                .map(move |plane| (domain, plane))
        });
        for (domain, plane) in planes {
            let width =
                u64::try_from(plane.row_elements).map_err(|_| "state plane width exceeds u64")?;
            let extents = vec![
                u64::try_from(domain.rows).map_err(|_| "state history rows exceed u64")?,
                1,
                width,
            ];
            for &rows in row_classes {
                let class = StateCopyGraphClass {
                    element: Element::dense(plane.dtype),
                    source_extents: extents.clone(),
                    destination_extents: extents.clone(),
                    map_rows: rows,
                    slab_rows: domain.slab_rows,
                };
                if !classes.contains(&class) {
                    classes.push(class);
                }
            }
        }
        if !state.recurrent_components.is_empty() {
            let banks = u64::try_from(
                state
                    .bank_capacity
                    .storage_total()
                    .map_err(|error| error.to_string())?,
            )
            .map_err(|_| "state bank count exceeds u64")?;
            for component in &state.recurrent_components {
                let width = component
                    .shape
                    .iter()
                    .try_fold(1u64, |size, &extent| {
                        size.checked_mul(u64::try_from(extent).ok()?)
                    })
                    .ok_or("state bank component width exceeds u64")?;
                let class = StateCopyGraphClass {
                    element: Element::dense(component.dtype),
                    source_extents: vec![banks, 1, width],
                    destination_extents: vec![banks, 1, width],
                    map_rows: 1,
                    slab_rows: state.bank_slab_banks()?,
                };
                if !classes.contains(&class) {
                    classes.push(class);
                }
            }
        }
    }
    Ok(classes)
}

impl NativeStateProgram {
    pub(crate) fn new(graphs: Rc<PreparedStateCopyGraphs>) -> Self {
        Self { graphs }
    }

    fn execute(
        &self,
        batch: &magnitude_batching::ValidatedStateBatch,
        work: &mut StateWork,
        state_graph_workspace: &mut NativeGraphWorkspaceLease,
    ) -> Result<(), SubmitError> {
        match work {
            StateWork::Copy(_) | StateWork::StoreCopy(_) => {
                let (planes, copies) = match work {
                    StateWork::Copy(advance) => {
                        let binding = advance.bindings();
                        (
                            binding
                                .history
                                .iter()
                                .map(|plane| (&plane.buffer, plane.slab_rows))
                                .collect::<Vec<_>>(),
                            binding.copies,
                        )
                    }
                    StateWork::StoreCopy(copy) => (
                        copy.planes()
                            .iter()
                            .map(|plane| (plane, copy.slab_rows()))
                            .collect::<Vec<_>>(),
                        copy.copies(),
                    ),
                    StateWork::CodecConversion(_) => unreachable!(),
                };
                if copies.is_empty() {
                    return Err(invalid("empty copy mapping"));
                }
                let words = |indices: &[usize]| {
                    indices
                        .iter()
                        .map(|&row| {
                            i32::try_from(row).map_err(|_| invalid("state row exceeds i32"))
                        })
                        .collect::<Result<Vec<_>, _>>()
                };
                let bytes = |indices: &[i32]| {
                    indices
                        .iter()
                        .flat_map(|row| row.to_le_bytes())
                        .collect::<Vec<_>>()
                };
                let class_rows = batch.class_rows();
                for copy in copies {
                    let &(plane, slab_rows) = planes
                        .get(copy.plane_index)
                        .ok_or_else(|| invalid("copy plane is outside pinned history"))?;
                    let shape = plane.extents();
                    let width = shape[1..]
                        .iter()
                        .try_fold(1u64, |n, extent| n.checked_mul(*extent))
                        .ok_or_else(|| invalid("state plane width overflow"))?;
                    let view = plane.reshape(&[shape[0], 1, width]).map_err(device)?;
                    let class = StateCopyGraphClass {
                        element: plane.element(),
                        source_extents: view.extents().to_vec(),
                        destination_extents: view.extents().to_vec(),
                        map_rows: class_rows as u64,
                        slab_rows,
                    };
                    let plan = self.graphs.plan(&class)?;
                    let bindings = self.graphs.bindings(&class, &view)?;
                    // Native copy bodies skip negative map entries. Padding
                    // must not launch concurrent writes to a real row.
                    let mut from = words(&copy.from)?;
                    from.resize(class_rows, -1);
                    let mut to = words(&copy.to)?;
                    to.resize(class_rows, -1);
                    self.graphs.run(
                        &class,
                        state_graph_workspace.slot_mut(),
                        bindings,
                        plan.new_outputs().map_err(device)?,
                        &bytes(&from),
                        &bytes(&to),
                    )?;
                }
                Ok(())
            }
            StateWork::CodecConversion(_) => Err(invalid(
                "codec conversion has no defined native numerical implementation",
            )),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StateCopyGraphClass {
    pub element: Element,
    pub source_extents: Vec<u64>,
    pub destination_extents: Vec<u64>,
    pub map_rows: u64,
    pub slab_rows: u32,
}

pub struct PreparedStateCopyGraphs {
    variants: Vec<PreparedStateCopyGraph>,
    family: NativeGraphFamily,
}

struct PreparedStateCopyGraph {
    class: StateCopyGraphClass,
    plan: NativeGraphPlan,
    rows: NativePort,
    from: NativePort,
    to: NativePort,
}

impl PreparedStateCopyGraphs {
    pub(crate) fn prepare(
        target_device: &Device,
        handles: &AttestedState,
        classes: impl IntoIterator<Item = StateCopyGraphClass>,
    ) -> Result<Self, SubmitError> {
        let classes = classes.into_iter().collect::<Vec<_>>();
        let (_, layouts) = certify_copy_family(target_device.backend(), &classes)
            .map_err(|error| invalid(error.to_string()))?;
        let mut variants = Vec::new();
        for (class, layout) in classes.into_iter().zip(layouts) {
            validate_class(&class).map_err(invalid)?;
            if variants
                .iter()
                .any(|variant: &PreparedStateCopyGraph| variant.class == class)
            {
                return Err(invalid("state copy graph class is duplicated"));
            }
            let kernel = handles
                .copies
                .iter()
                .find(|(element, _)| *element == class.element)
                .map(|(_, kernel)| kernel)
                .ok_or_else(|| invalid("state copy graph specialization is absent"))?;
            let (plan, rows, from, to) = copy_graph_topology(
                target_device.native_graph_with_layout(&layout),
                kernel,
                &class,
            )
            .map_err(device)?;
            variants.push(PreparedStateCopyGraph {
                class,
                plan,
                rows,
                from,
                to,
            });
        }
        if variants.is_empty() {
            return Err(invalid("state copy graph family has no classes"));
        }
        let plans = variants
            .iter()
            .map(|variant| variant.plan.clone())
            .collect::<Vec<_>>();
        let family = NativeGraphFamily::new(&plans).map_err(device)?;
        Ok(Self { variants, family })
    }

    pub fn workspace_bytes_max(&self) -> u64 {
        self.family.workspace_bytes()
    }

    pub fn output_bytes_max(&self) -> u64 {
        self.family.output_bytes()
    }

    pub fn family(&self) -> &NativeGraphFamily {
        &self.family
    }

    pub(crate) fn plan(
        &self,
        class: &StateCopyGraphClass,
    ) -> Result<&NativeGraphPlan, SubmitError> {
        Ok(&self.variant(class)?.plan)
    }

    pub(crate) fn bindings(
        &self,
        class: &StateCopyGraphClass,
        rows: &Tensor,
    ) -> Result<NativeGraphBindings, SubmitError> {
        let variant = self.variant(class)?;
        let mut bindings = variant.plan.bindings();
        bindings.set(&variant.rows, rows).map_err(device)?;
        Ok(bindings)
    }

    pub(crate) fn run(
        &self,
        class: &StateCopyGraphClass,
        slot: &mut NativeGraphFamilySlot,
        bindings: NativeGraphBindings,
        outputs: NativeGraphOutputs,
        from: &[u8],
        to: &[u8],
    ) -> Result<NativeGraphOutputs, SubmitError> {
        let variant = self.variant(class)?;
        let mut active = slot.activate(&variant.plan).map_err(device)?;
        active.write_input(&variant.from, from).map_err(device)?;
        active.write_input(&variant.to, to).map_err(device)?;
        active
            .attach(bindings, outputs)
            .and_then(super::run_graph)
            .map_err(device)
    }

    fn variant(&self, class: &StateCopyGraphClass) -> Result<&PreparedStateCopyGraph, SubmitError> {
        self.variants
            .iter()
            .find(|variant| variant.class == *class)
            .ok_or_else(|| invalid("state copy graph class was not prepared"))
    }
}

fn validate_class(class: &StateCopyGraphClass) -> Result<(), &'static str> {
    if class.map_rows == 0 {
        return Err("state copy graph has no mapped rows");
    }
    if class.source_extents.len() != 3 || class.destination_extents.len() != 3 {
        return Err("state copy graph requires rank-three planes");
    }
    if class.source_extents != class.destination_extents {
        return Err("state copy graph requires one in-place plane geometry");
    }
    Ok(())
}

fn copy_graph_topology<'a, G: GraphDraft + 'a>(
    graph: G,
    entry: G::Binding<'a, copy_rows::Entry>,
    class: &StateCopyGraphClass,
) -> Result<(G::Plan, NativePort, NativePort, NativePort), GraphError> {
    let (graph, rows, from, to) = copy_graph_draft(graph, entry, class)?;
    Ok((graph.seal()?, rows, from, to))
}

fn copy_graph_draft<'a, G: GraphDraft + 'a>(
    mut graph: G,
    entry: G::Binding<'a, copy_rows::Entry>,
    class: &StateCopyGraphClass,
) -> Result<(G, NativePort, NativePort, NativePort), GraphError> {
    let rows = graph.port(class.element, &class.source_extents)?;
    let dimensions = [
        ("N", class.map_rows),
        ("T", class.source_extents[0]),
        ("KV", class.source_extents[1]),
        ("W", class.source_extents[2]),
    ];
    let from = graph.input_for(entry, "from", &dimensions)?;
    let to = graph.input_for(entry, "to", &dimensions)?;
    let mut rows_tensor = rows.tensor().clone();
    graph.enqueue::<copy_rows::Entry>(
        entry,
        &dimensions,
        copy_rows::WorkflowArgs {
            rows: (&mut rows_tensor).into(),
            from: from.tensor().into(),
            to: to.tensor().into(),
            slab_rows: class.slab_rows,
        },
    )?;
    Ok((graph, rows, from, to))
}

pub(crate) fn checked_copy_family_storage(
    backend: BackendName,
    classes: impl IntoIterator<Item = StateCopyGraphClass>,
) -> Result<NativeGraphStorageBytes, GraphError> {
    let classes = classes.into_iter().collect::<Vec<_>>();
    certify_copy_family(backend, &classes).map(|(storage, _)| storage)
}

fn certify_copy_family(
    backend: BackendName,
    classes: &[StateCopyGraphClass],
) -> Result<(NativeGraphStorageBytes, Vec<NativeGraphLayout>), GraphError> {
    let mut family: Option<NativeGraphStorageBytes> = None;
    let mut layouts = vec![None; classes.len()];
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (index, class) in classes.iter().enumerate() {
        validate_class(class)?;
        if let Some(group) = groups.iter_mut().find(|group| {
            let first = &classes[group[0]];
            first.element == class.element
                && first.source_extents == class.source_extents
                && first.destination_extents == class.destination_extents
                && first.slab_rows == class.slab_rows
        }) {
            group.push(index);
        } else {
            groups.push(vec![index]);
        }
    }
    for group in groups {
        let first = &classes[group[0]];
        let elements = [("A", first.element)];
        let (graph, _, _, _) =
            copy_graph_draft(NativeGraphMetadata::new_template(backend), &elements, first)?;
        let layout = graph.seal_template().and_then(|template| {
            template.certify(&[NativeGraphClassSlice::new()
                .dimension("N", group.iter().map(|&index| classes[index].map_rows))])
        })?;
        let storage = layout.storage_bytes();
        match &mut family {
            Some(maximum) => {
                maximum.workspace = maximum.workspace.max(storage.workspace);
                maximum.output = maximum.output.max(storage.output);
                maximum.upload = maximum.upload.max(storage.upload);
            }
            None => family = Some(storage),
        }
        for &index in &group {
            layouts[index] = Some(layout.clone());
        }
    }
    Ok((
        family.ok_or("state copy graph family has no classes")?,
        layouts.into_iter().map(Option::unwrap).collect(),
    ))
}

impl StateProgram for NativeStateProgram {
    type Submission = ReadySubmission<StateLaunchCore, NativeGraphWorkspaceLease, ()>;
    fn submit(
        &mut self,
        mut launch: ValidatedStateLaunch,
    ) -> Result<Self::Submission, (SubmitError, ValidatedStateLaunch)> {
        let result = {
            let (batch, work, state_graph_workspace) = launch.execution_parts_mut();
            self.execute(batch, work, state_graph_workspace)
        };
        if let Err(error) = result {
            return Err((error, launch));
        }
        let (core, graph_workspace) = launch.into_submission_parts();
        Ok(ReadySubmission::new(core, graph_workspace, ()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_state::{BankCapacity, ComponentSpec};

    #[test]
    fn prepared_graph_copies_within_held_slabs() {
        let catalog = seismic::DeviceCatalog::discover().unwrap();
        let Ok(device) = catalog.open_backend(BackendName::Cpu) else {
            return;
        };
        let element = Element::f32();
        let kernel = copy_rows::native_for_device_with(
            &device,
            copy_rows::Elements { A: element },
            &seismic::NativeSpecialization::new(),
        )
        .unwrap();
        let handles = AttestedState {
            copies: vec![(element, kernel)],
            conditioning: None,
        };
        let class = StateCopyGraphClass {
            element,
            source_extents: vec![4, 1, 4],
            destination_extents: vec![4, 1, 4],
            map_rows: 2,
            slab_rows: 2,
        };
        let other = StateCopyGraphClass {
            map_rows: 1,
            ..class.clone()
        };
        let assessment =
            checked_copy_family_storage(BackendName::Cpu, [other.clone(), class.clone()]).unwrap();
        let graphs =
            PreparedStateCopyGraphs::prepare(&device, &handles, [other, class.clone()]).unwrap();
        assert_eq!(graphs.workspace_bytes_max(), assessment.workspace);
        assert_eq!(graphs.output_bytes_max(), assessment.output);
        let mut slabs = seismic::SlabTensor::new(
            &device,
            2,
            4,
            vec![seismic::SlabRegion {
                element,
                row_shape: vec![1, 4],
            }],
        )
        .unwrap();
        slabs.add_slab().unwrap();
        slabs.add_slab().unwrap();
        let source = [5.0f32, 6.0, 7.0, 8.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        slabs
            .region_rows(0, 3, 1)
            .unwrap()
            .write_from_host(&source)
            .unwrap();
        let view = slabs.logical_region(0).unwrap();
        let reshaped = view.reshape(&[4, 1, 4]).unwrap();
        let bindings = graphs.bindings(&class, &reshaped).unwrap();
        let outputs = graphs.plan(&class).unwrap().new_outputs().unwrap();
        let mut slot = graphs.family().new_slot(1).unwrap();
        graphs
            .run(
                &class,
                &mut slot,
                bindings,
                outputs,
                &[3_i32.to_le_bytes(), (-1_i32).to_le_bytes()].concat(),
                &[0_i32.to_le_bytes(), (-1_i32).to_le_bytes()].concat(),
            )
            .unwrap();
        assert_eq!(
            slabs.region_rows(0, 0, 1).unwrap().read_to_host().unwrap(),
            source
        );
    }

    #[test]
    fn state_copy_classes_include_recurrent_banks() {
        let plan = StateStorePlan {
            context_rows: 8,
            max_advance: 8,
            history: Vec::new(),
            domains: Vec::new(),
            recurrent_components: vec![ComponentSpec {
                shape: vec![4],
                dtype: seismic::DType::F32,
            }],
            bank_capacity: BankCapacity {
                active: 2,
                in_flight: 1,
                retained: 0,
            },
            recurrent_bank_bytes: 16,
            zero_seed_bytes: 16,
            recurrent_pool_bytes: 48,
        };
        let classes = state_copy_classes(&plan, None, &[1, 2]).unwrap();
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0].element, Element::f32());
        assert_eq!(classes[0].source_extents, vec![4, 1, 4]);
        assert_eq!(classes[0].map_rows, 1);
        assert_eq!(classes[0].slab_rows, plan.bank_slab_banks().unwrap());
    }

    #[test]
    fn checked_copy_family_uses_independent_storage_maxima() {
        let small = StateCopyGraphClass {
            element: Element::f32(),
            source_extents: vec![8, 1, 4],
            destination_extents: vec![8, 1, 4],
            map_rows: 1,
            slab_rows: 8,
        };
        let large = StateCopyGraphClass {
            map_rows: 2,
            ..small.clone()
        };
        let family =
            checked_copy_family_storage(BackendName::Cpu, [small.clone(), large.clone()]).unwrap();
        let one = checked_copy_family_storage(BackendName::Cpu, [small]).unwrap();
        let two = checked_copy_family_storage(BackendName::Cpu, [large]).unwrap();
        assert_eq!(family.workspace, one.workspace.max(two.workspace));
        assert_eq!(family.output, one.output.max(two.output));
        assert_eq!(family.upload, one.upload.max(two.upload));
        assert!(two.upload > one.upload);
        assert!(family.upload >= 2 * 2 * 4);
    }
}
