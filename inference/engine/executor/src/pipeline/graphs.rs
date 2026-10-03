//! Stage projection of the normal checked decoder graph builder.
use super::{PipelineRefusal, StageAssignment};
use crate::{
    programs::{
        graph::{
            RowForm,
            readout::{PreparedTargetReadoutGraphs, BoundTargetReadoutGraphs},
        },
        native_constants::ConstantTensors,
        native_target_graph::{
            certify_stage_blocks, certify_entry_layout, EntryTokens, PreparedTargetBlockGraph,
            PreparedTargetEntryGraph,
        },
    },
    AttestedPrograms, GraphSlots, ModelLoadPlan, NativeGraphCharge, ResidentWeight,
    StateResourcePlan,
};
use magnitude_family_contracts::{WeightRole, WeightKind, WeightScope};
use magnitude_state::{GlobalLayerId, StageLayerOrdinal};
use seismic::{BoundNativeGraphPlan, Device, NativeGraphFamily, NativeGraphStorageBytes, Tensor};
use std::collections::HashMap;

pub(crate) struct PreparedStageBlock {
    pub rows: u64,
    pub local: StageLayerOrdinal,
    pub global: GlobalLayerId,
    pub graph: PreparedTargetBlockGraph,
}
pub(crate) struct PreparedStageGraphs {
    pub entries: Vec<(u64, PreparedTargetEntryGraph)>,
    pub readout: Option<PreparedTargetReadoutGraphs>,
    pub readout_charge: Option<NativeGraphCharge>,
    pub blocks: Vec<PreparedStageBlock>,
    pub family: NativeGraphFamily,
    pub charge: NativeGraphCharge,
    pub constants_bytes: u64,
}
pub(crate) struct BoundStageGraphs {
    pub entries: Vec<(u64, PreparedTargetEntryGraph, BoundNativeGraphPlan)>,
    pub readout: Option<BoundTargetReadoutGraphs>,
    pub blocks: Vec<BoundStageBlock>,
    pub constants: Vec<Tensor>,
}

pub(crate) struct BoundStageBlock {
    pub prepared: PreparedStageBlock,
    pub bound: BoundNativeGraphPlan,
}
impl PreparedStageGraphs {
    pub fn prepare(
        device: &Device,
        programs: &AttestedPrograms,
        assignment: &StageAssignment,
        load: &ModelLoadPlan,
        state: &StateResourcePlan,
    ) -> Result<Self, PipelineRefusal> {
        let invalid = PipelineRefusal::Preparation;
        let limits = state.limits();
        if !matches!(limits.max_launch_rows, 1 | 2)
            || limits.max_launch_slots != 1
            || limits.max_selected_rows != 1
            || limits.exported_logits_rows > 1
            || limits.lookahead
        {
            return Err(PipelineRefusal::UnsupportedProfile);
        }
        if !programs.belongs_to(device) {
            return Err(PipelineRefusal::ForeignDevice);
        }
        let view = assignment.view();
        let plan = load
            .program_plan(assignment.definition(), magnitude_state::KvCodec::Dense)
            .map_err(|e| invalid(e.to_string()))?;
        let certificate =
            certify_stage_blocks(device.backend(), load, &view, state, plan.target(), limits)
                .map_err(|e| invalid(e.to_string()))?;
        let mut blocks = Vec::new();
        for rows in [1, 2]
            .into_iter()
            .filter(|&r| r <= limits.max_launch_rows as u64)
        {
            for entry in &certificate.blocks {
                let graph = PreparedTargetBlockGraph::prepare(
                    device,
                    programs.stage_block(entry.global).map_err(invalid)?,
                    None,
                    None,
                    load,
                    view.decoder(),
                    state,
                    entry.global.index() as usize,
                    rows,
                    1,
                    1,
                    &entry.layouts[&RowForm::of(rows)],
                    Some(&view),
                )
                .map_err(invalid)?;
                // Every graph role is an original assigned role, never a local
                // index rebased back onto an unrelated original block.
                for (weight, _) in &graph.weights {
                    if !assignment.owns_role(weight.role)? {
                        return Err(PipelineRefusal::InvalidWeightRole(weight.role));
                    }
                }
                blocks.push(PreparedStageBlock {
                    rows,
                    local: entry.local,
                    global: entry.global,
                    graph,
                });
            }
        }
        let mut entries = Vec::new();
        let mut expected_storage = certificate.resources.storage;
        if assignment.owns_entry() {
            let rows = [1, 2]
                .into_iter()
                .filter(|&r| r <= limits.max_launch_rows as u64)
                .collect::<Vec<_>>();
            let layout = certify_entry_layout(
                device.backend(),
                load,
                view.decoder(),
                &rows,
                EntryTokens::Uploaded,
            )
            .map_err(|e| invalid(e.to_string()))?;
            expected_storage.workspace = expected_storage
                .workspace
                .max(layout.storage_bytes().workspace);
            expected_storage.output = expected_storage.output.max(layout.storage_bytes().output);
            expected_storage.upload = expected_storage.upload.max(layout.storage_bytes().upload);
            for rows in rows {
                entries.push((
                    rows,
                    PreparedTargetEntryGraph::prepare(
                        device,
                        &programs.stage_target().embedding,
                        load,
                        view.decoder(),
                        rows,
                        EntryTokens::Uploaded,
                        &layout,
                    )
                    .map_err(invalid)?,
                ));
            }
        }
        let readout = if assignment.owns_readout() {
            Some(
                PreparedTargetReadoutGraphs::prepare(
                    device,
                    programs.stage_target(),
                    load,
                    view.decoder(),
                    limits,
                )
                .map_err(invalid)?,
            )
        } else {
            None
        };
        let readout_charge = readout
            .as_ref()
            .map(|graphs| {
                let family = graphs.family();
                NativeGraphCharge::from_checked(
                    NativeGraphStorageBytes {
                        workspace: family.workspace_bytes(),
                        output: family.output_bytes(),
                        upload: family.upload_bytes(),
                    },
                    1,
                    GraphSlots {
                        activations: 1,
                        output: 1,
                    },
                )
                .map_err(invalid)
            })
            .transpose()?;
        let plans = blocks
            .iter()
            .map(|b| b.graph.plan.clone())
            .chain(entries.iter().map(|(_, graph)| graph.plan.clone()))
            .collect::<Vec<_>>();
        let family = NativeGraphFamily::new(&plans).map_err(|e| invalid(e.to_string()))?;
        let storage = NativeGraphStorageBytes {
            workspace: family.workspace_bytes(),
            output: family.output_bytes(),
            upload: family.upload_bytes(),
        };
        if storage != expected_storage {
            return Err(invalid(
                "projected executable graph charge differs from checked certificate".into(),
            ));
        }
        let charge = NativeGraphCharge::from_checked(
            storage,
            1,
            GraphSlots {
                activations: if assignment.owns_entry() { 2 } else { 1 },
                output: if assignment.owns_entry() { 3 } else { 2 },
            },
        )
        .map_err(invalid)?;
        Ok(Self {
            entries,
            readout,
            readout_charge,
            blocks,
            family,
            charge,
            constants_bytes: certificate.resources.binding_constant_bytes,
        })
    }
    /// Bind only completed local imports and constants, under the startup heap
    /// claim. Seismic checks every fixed binding's exact device and contract.
    pub fn bind(
        self,
        device: &Device,
        weights: &HashMap<WeightRole, ResidentWeight>,
    ) -> Result<BoundStageGraphs, PipelineRefusal> {
        let invalid = PipelineRefusal::Preparation;
        let mut constants = ConstantTensors::new(device.clone());
        let mut entries = Vec::new();
        for (rows, graph) in self.entries {
            let role = WeightRole {
                scope: WeightScope::Target,
                kind: WeightKind::Embedding,
            };
            let weight = weights
                .get(&role)
                .ok_or(PipelineRefusal::InvalidWeightRole(role))?;
            if !weight.belongs_to(device) {
                return Err(PipelineRefusal::ForeignDevice);
            }
            let bound = graph
                .plan
                .bind_static(&[(&graph.table, weight.tensor())])
                .map_err(|e| invalid(e.to_string()))?;
            entries.push((rows, graph, bound));
        }
        let readout = self
            .readout
            .map(|graphs| {
                let get = |kind| {
                    let role = WeightRole {
                        scope: WeightScope::Target,
                        kind,
                    };
                    let weight = weights
                        .get(&role)
                        .ok_or(PipelineRefusal::InvalidWeightRole(role))?;
                    if !weight.belongs_to(device) {
                        return Err(PipelineRefusal::ForeignDevice);
                    }
                    Ok(weight)
                };
                graphs
                    .bind_parts(get(WeightKind::OutputNorm)?, get(WeightKind::Output)?, None)
                    .map_err(invalid)
            })
            .transpose()?;
        let mut bound = Vec::new();
        for prepared in self.blocks {
            let fixed_weights = prepared
                .graph
                .weights
                .iter()
                .map(|(port, _)| {
                    let weight = weights
                        .get(&port.role)
                        .ok_or(PipelineRefusal::InvalidWeightRole(port.role))?;
                    if !weight.belongs_to(device) {
                        return Err(PipelineRefusal::ForeignDevice);
                    }
                    port.part
                        .of(weight)
                        .cloned()
                        .map_err(|e| invalid(e.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let plan = prepared
                .graph
                .bind_resolved_weights(&fixed_weights, &mut constants)
                .map_err(invalid)?;
            bound.push(BoundStageBlock {
                prepared,
                bound: plan,
            });
        }
        let constants = constants.into_tensors();
        let bytes = constants.iter().try_fold(0u64, |total, t| {
            total
                .checked_add(t.storage_bytes())
                .ok_or_else(|| invalid("constant charge overflows".into()))
        })?;
        if bytes != self.constants_bytes {
            return Err(invalid(
                "bound constants differ from projected charge".into(),
            ));
        }
        Ok(BoundStageGraphs {
            entries,
            readout,
            blocks: bound,
            constants,
        })
    }
}
