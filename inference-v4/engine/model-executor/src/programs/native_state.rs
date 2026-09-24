//! Native state maintenance over validated row maps and pinned state planes.

use super::{ReadySubmission, StateProgram};
use crate::{
    DeviceError, InvariantError, NativeGraphWorkspaceLease, StateLaunchCore, StateWork,
    SubmitError, ValidatedStateLaunch, native::AttestedState,
};
use magnitude_model_kernels::copy_rows;
use magnitude_model_state::StateStore;
use seismic::{
    Device, Element, NativeGraphBindings, NativeGraphFamily, NativeGraphFamilySlot,
    NativeGraphOutputs, NativeGraphPlan, NativePort, Tensor,
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
    repair: Option<(super::native_target::NativeTargetProgram, Rc<StateStore>)>,
    graphs: Rc<PreparedStateCopyGraphs>,
}

impl NativeStateProgram {
    pub(crate) fn new(graphs: Rc<PreparedStateCopyGraphs>) -> Self {
        Self {
            repair: None,
            graphs,
        }
    }
    pub(crate) fn with_repair(
        mut self,
        target: super::native_target::NativeTargetProgram,
        store: Rc<StateStore>,
    ) -> Self {
        self.repair = Some((target, store));
        self
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
                let first = binding
                    .copies
                    .first()
                    .ok_or_else(|| invalid("empty copy mapping"))?;
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
            StateWork::RecurrentRepair {
                advance,
                conditioning,
                conditioning_slices,
                graph_workspace,
                graph_outputs,
            } => {
                let (target, store) = self
                    .repair
                    .as_ref()
                    .ok_or_else(|| invalid("repair target program is not bound"))?;
                let replay = batch
                    .replay()
                    .ok_or_else(|| invalid("repair replay controls are absent"))?;
                let history = store.history_planes().map_err(device)?;
                target.execute_repair(
                    replay,
                    advance,
                    &history,
                    conditioning.as_ref(),
                    conditioning_slices,
                    graph_workspace,
                    graph_outputs,
                )
            }
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
            if class.map_rows == 0 {
                return Err(invalid("state copy graph has no mapped rows"));
            }
            if class.source_extents.len() != 3 || class.destination_extents.len() != 3 {
                return Err(invalid("state copy graph requires rank-three planes"));
            }
            if class.source_extents[1..] != class.destination_extents[1..] {
                return Err(invalid(
                    "state copy graph source and destination plane geometry differ",
                ));
            }
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
            let mut graph = target_device.native_graph();
            let source = graph
                .port(class.element, &class.source_extents)
                .map_err(device)?;
            let destination = graph
                .port(class.element, &class.destination_extents)
                .map_err(device)?;
            let from = graph
                .input_for(
                    kernel,
                    "from",
                    &[
                        ("N", class.map_rows),
                        ("TS", class.source_extents[0]),
                        ("TD", class.destination_extents[0]),
                        ("KV", class.source_extents[1]),
                        ("W", class.source_extents[2]),
                    ],
                )
                .map_err(device)?;
            let to = graph
                .input_for(
                    kernel,
                    "to",
                    &[
                        ("N", class.map_rows),
                        ("TS", class.source_extents[0]),
                        ("TD", class.destination_extents[0]),
                        ("KV", class.source_extents[1]),
                        ("W", class.source_extents[2]),
                    ],
                )
                .map_err(device)?;
            let mut destination_tensor = destination.tensor().clone();
            graph
                .enqueue(
                    kernel,
                    copy_rows::WorkflowArgs {
                        src: source.tensor().into(),
                        dst: (&mut destination_tensor).into(),
                        from: from.tensor().into(),
                        to: to.tensor().into(),
                    },
                )
                .map_err(device)?;
            let plan = graph.seal().map_err(device)?;
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

    pub(crate) fn plans(&self) -> impl Iterator<Item = (&StateCopyGraphClass, &NativeGraphPlan)> {
        self.variants
            .iter()
            .map(|variant| (&variant.class, &variant.plan))
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
