//! Native state maintenance over validated row maps and pinned state planes.

use super::{ReadySubmission, StateProgram};
use crate::{
    native::AttestedState, programs::graph::draft::GraphDraft, DeviceError, InvariantError,
    NativeGraphWorkspaceLease, StateLaunchCore, StateStorePlan, StateWork, SubmitError,
    ValidatedStateLaunch,
};
use magnitude_model_kernels::copy_rows;
use seismic::{
    BackendName, Device, Element, NativeGraphBindings, NativeGraphFamily, NativeGraphFamilySlot,
    NativeGraphMetadata, NativeGraphOutputs, NativeGraphPlan, NativeGraphStorageBytes, NativePort,
    Tensor,
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
        for component in &state.history_components {
            for plane in component.planes() {
                let width = u64::try_from(plane.row_elements)
                    .map_err(|_| "state plane width exceeds u64")?;
                let extents = vec![
                    u64::try_from(state.history_rows)
                        .map_err(|_| "state history rows exceed u64")?,
                    1,
                    width,
                ];
                for &rows in row_classes {
                    let class = StateCopyGraphClass {
                        element: Element::dense(plane.dtype),
                        source_extents: extents.clone(),
                        destination_extents: extents.clone(),
                        map_rows: rows,
                    };
                    if !classes.contains(&class) {
                        classes.push(class);
                    }
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
        batch: &magnitude_model_batching::ValidatedStateBatch,
        work: &mut StateWork,
        state_graph_workspace: &mut NativeGraphWorkspaceLease,
    ) -> Result<(), SubmitError> {
        match work {
            StateWork::Copy(advance) => {
                let binding = advance.bindings();
                if binding.copies.is_empty() {
                    return Err(invalid("empty copy mapping"));
                }
                let bytes = |indices: &[usize]| {
                    indices
                        .iter()
                        .map(|&row| {
                            i32::try_from(row)
                                .map(|value| value.to_le_bytes())
                                .map_err(|_| invalid("state row exceeds i32"))
                        })
                        .collect::<Result<Vec<_>, _>>()
                        .map(|words| words.into_iter().flatten().collect::<Vec<_>>())
                };
                let class_rows = batch.class_rows();
                for copy in binding.copies {
                    let plane = binding
                        .history
                        .get(copy.plane_index)
                        .ok_or_else(|| invalid("copy plane is outside pinned history"))?;
                    let shape = plane.buffer.extents();
                    let width = shape[1..]
                        .iter()
                        .try_fold(1u64, |n, extent| n.checked_mul(*extent))
                        .ok_or_else(|| invalid("state plane width overflow"))?;
                    let view = plane
                        .buffer
                        .reshape(&[shape[0], 1, width])
                        .map_err(device)?;
                    let class = StateCopyGraphClass {
                        element: plane.buffer.element(),
                        source_extents: view.extents().to_vec(),
                        destination_extents: view.extents().to_vec(),
                        map_rows: class_rows as u64,
                    };
                    let plan = self.graphs.plan(&class)?;
                    let bindings = self.graphs.bindings(&class, &view, &view)?;
                    // Padding lanes repeat the last real pair: an identical
                    // write, where a pad of row 0 would race a real copy
                    // into row 0.
                    let (Some(&last_from), Some(&last_to)) = (copy.from.last(), copy.to.last())
                    else {
                        return Err(invalid("empty copy mapping"));
                    };
                    let mut from = copy.from.clone();
                    from.resize(class_rows, last_from);
                    let mut to = copy.to.clone();
                    to.resize(class_rows, last_to);
                    self.graphs.run(
                        &class,
                        state_graph_workspace.slot_mut(),
                        bindings,
                        plan.new_outputs().map_err(device)?,
                        &bytes(&from)?,
                        &bytes(&to)?,
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
}

pub struct PreparedStateCopyGraphs {
    variants: Vec<PreparedStateCopyGraph>,
    family: NativeGraphFamily,
}

struct PreparedStateCopyGraph {
    class: StateCopyGraphClass,
    plan: NativeGraphPlan,
    source: NativePort,
    destination: NativePort,
    from: NativePort,
    to: NativePort,
}

impl PreparedStateCopyGraphs {
    pub(crate) fn prepare(
        target_device: &Device,
        handles: &AttestedState,
        classes: impl IntoIterator<Item = StateCopyGraphClass>,
    ) -> Result<Self, SubmitError> {
        let mut variants = Vec::new();
        for class in classes {
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
            let (plan, source, destination, from, to) =
                copy_graph_topology(target_device.native_graph(), kernel, &class)
                    .map_err(device)?;
            variants.push(PreparedStateCopyGraph {
                class,
                plan,
                source,
                destination,
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
        source: &Tensor,
        destination: &Tensor,
    ) -> Result<NativeGraphBindings, SubmitError> {
        let variant = self.variant(class)?;
        let mut bindings = variant.plan.bindings();
        bindings.set(&variant.source, source).map_err(device)?;
        bindings
            .set(&variant.destination, destination)
            .map_err(device)?;
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
    if class.source_extents[1..] != class.destination_extents[1..] {
        return Err("state copy graph source and destination plane geometry differ");
    }
    Ok(())
}

fn copy_graph_topology<'a, G: GraphDraft + 'a>(
    mut graph: G,
    entry: G::Binding<'a, copy_rows::Entry>,
    class: &StateCopyGraphClass,
) -> Result<(G::Plan, NativePort, NativePort, NativePort, NativePort), String> {
    let source = graph.port(class.element, &class.source_extents)?;
    let destination = graph.port(class.element, &class.destination_extents)?;
    let dimensions = [
        ("N", class.map_rows),
        ("TS", class.source_extents[0]),
        ("TD", class.destination_extents[0]),
        ("KV", class.source_extents[1]),
        ("W", class.source_extents[2]),
    ];
    let from = graph.input_for(entry, "from", &dimensions)?;
    let to = graph.input_for(entry, "to", &dimensions)?;
    let mut destination_tensor = destination.tensor().clone();
    graph.enqueue::<copy_rows::Entry>(
        entry,
        &dimensions,
        copy_rows::WorkflowArgs {
            src: source.tensor().into(),
            dst: (&mut destination_tensor).into(),
            from: from.tensor().into(),
            to: to.tensor().into(),
        },
    )?;
    Ok((graph.seal()?, source, destination, from, to))
}

pub(crate) fn checked_copy_family_storage(
    backend: BackendName,
    classes: impl IntoIterator<Item = StateCopyGraphClass>,
) -> Result<NativeGraphStorageBytes, String> {
    let mut family: Option<NativeGraphStorageBytes> = None;
    for class in classes {
        validate_class(&class).map_err(str::to_owned)?;
        let elements = [("A", class.element)];
        let (storage, _, _, _, _) =
            copy_graph_topology(NativeGraphMetadata::new(backend), &elements, &class)?;
        match &mut family {
            Some(maximum) => {
                maximum.workspace = maximum.workspace.max(storage.workspace);
                maximum.output = maximum.output.max(storage.output);
                maximum.upload = maximum.upload.max(storage.upload);
            }
            None => family = Some(storage),
        }
    }
    family.ok_or_else(|| "state copy graph family has no classes".into())
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

    #[test]
    fn checked_copy_family_uses_independent_storage_maxima() {
        let small = StateCopyGraphClass {
            element: Element::f32(),
            source_extents: vec![8, 1, 4],
            destination_extents: vec![8, 1, 4],
            map_rows: 1,
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
