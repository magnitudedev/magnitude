//! Ordered native vision tower over one validated image and pooled tensors.

use super::{ReadySubmission, VisionProgram};
use crate::{
    native::AttestedVision, programs::graph::draft::GraphDraft, DeviceError, GraphOutputTensor,
    InvariantError, ModelLoadPlan, NativeGraphOutputLease, NativeGraphWorkspaceLease,
    ResidentVision, ResidentWeight, SubmitError, ValidatedVisionLaunch, VisionLaunchCore,
    VisionProgramPlan, WeightPlan,
};
use magnitude_model_contracts::{VisionGeometry, WeightKind, WeightRole, WeightScope};
use magnitude_model_kernels::{qwen_vision_block, qwen_vision_merger, qwen_vision_stem};
use seismic::{
    BackendName, BoundNativeGraphPlan, Device, NativeGraphFamily, NativeGraphFamilySlot,
    NativeGraphMetadata, NativeGraphPlan, NativeGraphStorageBytes, NativePort, WorkflowTensor,
};
use std::rc::Rc;

fn invalid(detail: impl Into<String>) -> SubmitError {
    SubmitError::Invariant(InvariantError {
        context: "native vision program",
        detail: detail.into(),
    })
}
fn device(error: impl ToString) -> SubmitError {
    SubmitError::Device(DeviceError::Execution(error.to_string()))
}
fn planned_vision_weight_plan(
    load: &ModelLoadPlan,
    role: WeightRole,
) -> Result<&WeightPlan, SubmitError> {
    load.weights()
        .find(|plan| plan.role == role)
        .ok_or_else(|| invalid(format!("planned vision weight {role:?} is absent")))
}

fn planned_vision_weight<G: GraphDraft>(
    graph: &mut G,
    load: &ModelLoadPlan,
    role: WeightRole,
    weights: &mut Vec<(WeightRole, NativePort)>,
) -> Result<WorkflowTensor, SubmitError> {
    let plan = planned_vision_weight_plan(load, role)?;
    let port = graph.port(plan.resident, &plan.shape).map_err(device)?;
    let tensor = port.tensor().clone();
    weights.push((role, port));
    Ok(tensor)
}

struct VisionEntries<'a, G: GraphDraft + 'a> {
    stem: G::Binding<'a, qwen_vision_stem::Entry>,
    blocks: Vec<G::Binding<'a, qwen_vision_block::Entry>>,
    merger: G::Binding<'a, qwen_vision_merger::Entry>,
}

struct VisionGraphPorts {
    pixels: NativePort,
    positions: NativePort,
    rotation: NativePort,
    coordinates: NativePort,
    weights: Vec<(WeightRole, NativePort)>,
    features: WorkflowTensor,
}

fn vision_graph_topology<'a, G: GraphDraft + 'a>(
    mut graph: G,
    entries: VisionEntries<'a, G>,
    load: &ModelLoadPlan,
    geometry: &VisionGeometry,
    decoder_hidden: u64,
    rows: u64,
) -> Result<(G::Plan, VisionGraphPorts), SubmitError> {
    let merge = geometry
        .merge
        .checked_mul(geometry.merge)
        .ok_or_else(|| invalid("vision merge area overflow"))?;
    let four_heads = geometry
        .heads
        .checked_mul(4)
        .ok_or_else(|| invalid("vision attention width overflow"))?;
    if merge == 0 || four_heads == 0 || geometry.hidden % four_heads != 0 {
        return Err(invalid("vision graph geometry is not divisible"));
    }
    if rows == 0 || rows % merge != 0 {
        return Err(invalid(
            "vision graph patch class is not a positive merge-area multiple",
        ));
    }
    if entries.blocks.is_empty() {
        return Err(invalid("vision graph has no block entry"));
    }
    let head_width = geometry.hidden / four_heads;
    let merger_hidden = merge
        .checked_mul(geometry.hidden)
        .ok_or_else(|| invalid("vision merger hidden width overflow"))?;
    let merger_output = planned_vision_weight_plan(
        load,
        WeightRole {
            scope: WeightScope::Vision,
            kind: WeightKind::MergerOutput,
        },
    )?;
    let merger_bias = planned_vision_weight_plan(
        load,
        WeightRole {
            scope: WeightScope::Vision,
            kind: WeightKind::MergerOutputBias,
        },
    )?;
    if merger_output.shape != [decoder_hidden, merger_hidden]
        || merger_bias.shape != [decoder_hidden]
    {
        return Err(invalid(
            "vision merger output does not match the decoder hidden width",
        ));
    }

    let mut weights = Vec::new();
    macro_rules! weight {
        ($scope:expr, $kind:expr) => {
            planned_vision_weight(
                &mut graph,
                load,
                WeightRole {
                    scope: $scope,
                    kind: $kind,
                },
                &mut weights,
            )?
        };
    }
    let position = planned_vision_weight_plan(
        load,
        WeightRole {
            scope: WeightScope::Vision,
            kind: WeightKind::PositionEmbedding,
        },
    )?;
    let stem_dims = [
        ("M", rows),
        ("C", geometry.channels),
        ("P", geometry.patch),
        ("H", geometry.hidden),
        ("L", position.shape[0]),
    ];
    let pixels = graph
        .input_for(entries.stem, "pixels", &stem_dims)
        .map_err(device)?;
    let positions = graph
        .input_for(entries.stem, "indices", &stem_dims)
        .map_err(device)?;
    let rotation = graph
        .input_for(entries.stem, "coefficients", &stem_dims)
        .map_err(device)?;
    let temporal_weight_0 = weight!(WeightScope::VisionPatch(0), WeightKind::PatchEmbedding);
    let temporal_weight_1 = weight!(WeightScope::VisionPatch(1), WeightKind::PatchEmbedding);
    let bias = weight!(WeightScope::Vision, WeightKind::PatchBias);
    let table = weight!(WeightScope::Vision, WeightKind::PositionEmbedding);
    let stem = graph
        .enqueue::<qwen_vision_stem::Entry>(
            entries.stem,
            &stem_dims,
            qwen_vision_stem::WorkflowArgs {
                pixels: pixels.tensor().into(),
                temporal_weight_0: (&temporal_weight_0).into(),
                temporal_weight_1: (&temporal_weight_1).into(),
                bias: (&bias).into(),
                table: (&table).into(),
                indices: positions.tensor().into(),
                coefficients: rotation.tensor().into(),
            },
        )
        .map_err(device)?
        .value;
    let first_intermediate = planned_vision_weight_plan(
        load,
        WeightRole {
            scope: WeightScope::VisionBlock(0),
            kind: WeightKind::FeedForwardUpBias,
        },
    )?
    .shape[0];
    let block_dims = [
        ("M", rows),
        ("H", geometry.heads),
        ("P", head_width),
        ("F", first_intermediate),
    ];
    let coordinates = graph
        .input_for(entries.blocks[0], "coordinates", &block_dims)
        .map_err(device)?;
    let mut hidden = stem;
    for (index, entry) in entries.blocks.iter().copied().enumerate() {
        let scope = WeightScope::VisionBlock(index as u32);
        let intermediate = planned_vision_weight_plan(
            load,
            WeightRole {
                scope,
                kind: WeightKind::FeedForwardUpBias,
            },
        )?
        .shape[0];
        let dimensions = [
            ("M", rows),
            ("H", geometry.heads),
            ("P", head_width),
            ("F", intermediate),
        ];
        let norm1_weight = weight!(scope, WeightKind::InputNormWeight);
        let norm1_bias = weight!(scope, WeightKind::InputNormBias);
        let qkv_weight = weight!(scope, WeightKind::FusedQkvWeight);
        let qkv_bias = weight!(scope, WeightKind::FusedQkvBias);
        let projection_weight = weight!(scope, WeightKind::AttentionOutput);
        let projection_bias = weight!(scope, WeightKind::AttentionOutputBias);
        let norm2_weight = weight!(scope, WeightKind::FeedForwardNormWeight);
        let norm2_bias = weight!(scope, WeightKind::FeedForwardNormBias);
        let up_weight = weight!(scope, WeightKind::DenseUp);
        let up_bias = weight!(scope, WeightKind::FeedForwardUpBias);
        let down_weight = weight!(scope, WeightKind::DenseDown);
        let down_bias = weight!(scope, WeightKind::FeedForwardDownBias);
        let hidden_view = hidden.reshape(&[rows, geometry.heads, 4, head_width]);
        hidden = graph
            .enqueue::<qwen_vision_block::Entry>(
                entry,
                &dimensions,
                qwen_vision_block::WorkflowArgs {
                    hidden: (&hidden_view).into(),
                    coordinates: coordinates.tensor().into(),
                    norm1_weight: (&norm1_weight).into(),
                    norm1_bias: (&norm1_bias).into(),
                    qkv_weight: (&qkv_weight).into(),
                    qkv_bias: (&qkv_bias).into(),
                    projection_weight: (&projection_weight).into(),
                    projection_bias: (&projection_bias).into(),
                    norm2_weight: (&norm2_weight).into(),
                    norm2_bias: (&norm2_bias).into(),
                    up_weight: (&up_weight).into(),
                    up_bias: (&up_bias).into(),
                    down_weight: (&down_weight).into(),
                    down_bias: (&down_bias).into(),
                    epsilon: geometry.epsilon as f32,
                },
            )
            .map_err(device)?
            .value;
    }
    let norm_weight = weight!(WeightScope::Vision, WeightKind::NormWeight);
    let norm_bias = weight!(WeightScope::Vision, WeightKind::NormBias);
    let up_weight = weight!(WeightScope::Vision, WeightKind::MergerHidden);
    let up_bias = weight!(WeightScope::Vision, WeightKind::MergerHiddenBias);
    let down_weight = weight!(WeightScope::Vision, WeightKind::MergerOutput);
    let down_bias = weight!(WeightScope::Vision, WeightKind::MergerOutputBias);
    let merger_dims = [
        ("M", rows / merge),
        ("G", merge),
        ("H", geometry.hidden),
        ("D", decoder_hidden),
    ];
    let features = graph
        .enqueue::<qwen_vision_merger::Entry>(
            entries.merger,
            &merger_dims,
            qwen_vision_merger::WorkflowArgs {
                hidden: (&hidden).into(),
                norm_weight: (&norm_weight).into(),
                norm_bias: (&norm_bias).into(),
                up_weight: (&up_weight).into(),
                up_bias: (&up_bias).into(),
                down_weight: (&down_weight).into(),
                down_bias: (&down_bias).into(),
                epsilon: geometry.epsilon as f32,
            },
        )
        .map_err(device)?
        .value;
    graph.export(&features).map_err(device)?;
    let plan = graph.seal().map_err(device)?;
    Ok((
        plan,
        VisionGraphPorts {
            pixels,
            positions,
            rotation,
            coordinates,
            weights,
            features,
        },
    ))
}

pub(crate) fn checked_vision_family_storage(
    backend: BackendName,
    load: &ModelLoadPlan,
    geometry: &VisionGeometry,
    decoder_hidden: u64,
    plan: &VisionProgramPlan,
    patch_rows: impl IntoIterator<Item = u64>,
) -> Result<NativeGraphStorageBytes, String> {
    let patch = plan.patch();
    let stem_elements = [
        ("W0", patch.temporal_weight_0),
        ("W1", patch.temporal_weight_1),
        ("B", patch.bias),
        ("PE", patch.position),
    ];
    let block_elements = plan
        .blocks()
        .iter()
        .map(|block| {
            vec![
                ("A", block.activation),
                ("N1W", block.input_norm_weight),
                ("N1B", block.input_norm_bias),
                ("QW", block.qkv_weight),
                ("QB", block.qkv_bias),
                ("PW", block.attention_output),
                ("PB", block.attention_output_bias),
                ("N2W", block.feedforward_norm_weight),
                ("N2B", block.feedforward_norm_bias),
                ("UW", block.up),
                ("UB", block.up_bias),
                ("DW", block.down),
                ("DB", block.down_bias),
            ]
        })
        .collect::<Vec<_>>();
    let merger = plan.merger();
    let merger_elements = [
        ("A", merger.activation),
        ("NW", merger.output_norm_weight),
        ("NB", merger.output_norm_bias),
        ("UW", merger.hidden),
        ("UB", merger.hidden_bias),
        ("DW", merger.output),
        ("DB", merger.output_bias),
    ];
    let mut seen = Vec::new();
    let mut family: Option<NativeGraphStorageBytes> = None;
    for rows in patch_rows {
        if seen.contains(&rows) {
            return Err("vision graph patch class is duplicated".into());
        }
        seen.push(rows);
        let entries = VisionEntries {
            stem: &stem_elements[..],
            blocks: block_elements.iter().map(Vec::as_slice).collect(),
            merger: &merger_elements[..],
        };
        let (storage, _) = vision_graph_topology(
            NativeGraphMetadata::new(backend),
            entries,
            load,
            geometry,
            decoder_hidden,
            rows,
        )
        .map_err(|error| error.to_string())?;
        match &mut family {
            Some(maximum) => {
                maximum.workspace = maximum.workspace.max(storage.workspace);
                maximum.output = maximum.output.max(storage.output);
                maximum.upload = maximum.upload.max(storage.upload);
            }
            None => family = Some(storage),
        }
    }
    family.ok_or_else(|| "vision graph family has no exact patch class".into())
}
pub struct NativeVisionProgram {
    graphs: BoundVisionGraphs,
}

impl NativeVisionProgram {
    pub(crate) fn new(graphs: BoundVisionGraphs) -> Self {
        Self { graphs }
    }

    fn execute_graph(
        &self,
        core: &VisionLaunchCore,
        workspace: &mut NativeGraphWorkspaceLease,
        output: &mut Option<NativeGraphOutputLease>,
    ) -> Result<GraphOutputTensor, SubmitError> {
        let input = core.batch().input();
        let rows = u64::try_from(core.batch().patch_rows())
            .map_err(|_| invalid("vision patch rows exceed u64"))?;
        let spatial = input.spatial();
        let indices = spatial.interpolation_indices();
        let positions = (0..rows as usize)
            .flat_map(|row| (0..4).map(move |plane| indices[plane][row]))
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let coefficients = spatial.interpolation_coefficients();
        let rotation = (0..rows as usize)
            .flat_map(|row| (0..4).map(move |plane| coefficients[plane][row]))
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let coordinates = spatial
            .attention_coordinates()
            .iter()
            .flat_map(|pair| pair.iter().copied())
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let (prepared, bound) = self.graphs.class(rows)?;
        let output = output
            .take()
            .ok_or_else(|| invalid("vision graph output lease is absent"))?;
        Ok(prepared
            .run(
                rows,
                bound,
                workspace.slot_mut(),
                output,
                VisionGraphUploads {
                    pixels: input.pixels().data(),
                    positions: &positions,
                    rotation: &rotation,
                    coordinates: &coordinates,
                },
            )?
            .features)
    }
}

struct PreparedVisionGraph {
    patch_rows: u64,
    plan: NativeGraphPlan,
    pixels: NativePort,
    positions: NativePort,
    rotation: NativePort,
    coordinates: NativePort,
    weights: Vec<(WeightRole, NativePort)>,
    features: WorkflowTensor,
}

pub struct PreparedVisionGraphs {
    variants: Vec<PreparedVisionGraph>,
    family: NativeGraphFamily,
}

pub(crate) struct BoundVisionGraphs {
    prepared: Rc<PreparedVisionGraphs>,
    variants: Vec<BoundNativeGraphPlan>,
}

pub(crate) struct VisionGraphUploads<'a> {
    pub pixels: &'a [u8],
    pub positions: &'a [u8],
    pub rotation: &'a [u8],
    pub coordinates: &'a [u8],
}

pub(crate) struct VisionGraphResult {
    pub features: GraphOutputTensor,
}

impl PreparedVisionGraphs {
    pub(crate) fn prepare_exact_classes(
        target_device: &Device,
        handles: &AttestedVision,
        load: &ModelLoadPlan,
        geometry: &VisionGeometry,
        decoder_hidden: u64,
        patch_rows: impl IntoIterator<Item = u64>,
    ) -> Result<Self, SubmitError> {
        let mut variants = Vec::new();
        for rows in patch_rows {
            if variants
                .iter()
                .any(|variant: &PreparedVisionGraph| variant.patch_rows == rows)
            {
                return Err(invalid("vision graph patch class is duplicated"));
            }
            let entries = VisionEntries {
                stem: &handles.stem,
                blocks: handles.blocks.iter().collect(),
                merger: &handles.merger,
            };
            let (plan, ports) = vision_graph_topology(
                target_device.native_graph(),
                entries,
                load,
                geometry,
                decoder_hidden,
                rows,
            )?;
            variants.push(PreparedVisionGraph {
                patch_rows: rows,
                plan,
                pixels: ports.pixels,
                positions: ports.positions,
                rotation: ports.rotation,
                coordinates: ports.coordinates,
                weights: ports.weights,
                features: ports.features,
            });
        }
        if variants.is_empty() {
            return Err(invalid("vision graph family has no exact patch class"));
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

    pub(crate) fn bind_weights(
        self: &Rc<Self>,
        resident: &ResidentVision,
    ) -> Result<BoundVisionGraphs, SubmitError> {
        let variants = self
            .variants
            .iter()
            .map(|variant| {
                let fixed = variant
                    .weights
                    .iter()
                    .map(|(role, port)| {
                        Ok((port, resident_vision_weight(resident, *role)?.tensor()))
                    })
                    .collect::<Result<Vec<_>, SubmitError>>()?;
                variant.plan.bind_static(&fixed).map_err(device)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(BoundVisionGraphs {
            prepared: self.clone(),
            variants,
        })
    }

    pub(crate) fn run(
        &self,
        patch_rows: u64,
        bound: &BoundNativeGraphPlan,
        slot: &mut NativeGraphFamilySlot,
        mut output: NativeGraphOutputLease,
        uploads: VisionGraphUploads<'_>,
    ) -> Result<VisionGraphResult, SubmitError> {
        let variant = self
            .variants
            .iter()
            .find(|variant| variant.patch_rows == patch_rows)
            .ok_or_else(|| invalid("vision graph patch class was not prepared"))?;
        let bindings = bound.bindings();
        let mut active = slot.activate(&variant.plan).map_err(device)?;
        active
            .write_input(&variant.pixels, uploads.pixels)
            .map_err(device)?;
        active
            .write_input(&variant.positions, uploads.positions)
            .map_err(device)?;
        active
            .write_input(&variant.rotation, uploads.rotation)
            .map_err(device)?;
        active
            .write_input(&variant.coordinates, uploads.coordinates)
            .map_err(device)?;
        let outputs = output
            .activate(&variant.plan)
            .map_err(SubmitError::Invariant)?;
        let outputs = active
            .attach(bindings, outputs)
            .and_then(super::run_graph)
            .map_err(device)?;
        let owner = output.publish(outputs);
        let features = owner
            .tensor(&variant.features)
            .ok_or_else(|| invalid("vision graph omitted retained features"))?;
        Ok(VisionGraphResult { features })
    }
}

impl BoundVisionGraphs {
    pub(crate) fn class(
        &self,
        patch_rows: u64,
    ) -> Result<(&PreparedVisionGraphs, &BoundNativeGraphPlan), SubmitError> {
        let index = self
            .prepared
            .variants
            .iter()
            .position(|variant| variant.patch_rows == patch_rows)
            .ok_or_else(|| invalid("vision graph patch class was not prepared"))?;
        Ok((&self.prepared, &self.variants[index]))
    }
}

fn resident_vision_weight(
    resident: &ResidentVision,
    role: WeightRole,
) -> Result<&ResidentWeight, SubmitError> {
    let weight = match (role.scope, role.kind) {
        (WeightScope::VisionPatch(index), WeightKind::PatchEmbedding) => resident
            .patch_embeddings
            .get(index as usize)
            .ok_or_else(|| invalid("resident vision patch weight is absent"))?,
        (WeightScope::VisionBlock(index), kind) => {
            let block = resident
                .blocks
                .get(index as usize)
                .ok_or_else(|| invalid("resident vision block is absent"))?;
            match kind {
                WeightKind::InputNormWeight => &block.input_norm.weight,
                WeightKind::InputNormBias => &block.input_norm.bias,
                WeightKind::FusedQkvWeight => &block.attention.qkv.weight,
                WeightKind::FusedQkvBias => &block.attention.qkv.bias,
                WeightKind::AttentionOutput => &block.attention.output,
                WeightKind::AttentionOutputBias => &block.attention.output_bias,
                WeightKind::FeedForwardNormWeight => &block.feedforward_norm.weight,
                WeightKind::FeedForwardNormBias => &block.feedforward_norm.bias,
                WeightKind::DenseUp => &block.feedforward.up,
                WeightKind::FeedForwardUpBias => &block.feedforward.up_bias,
                WeightKind::DenseDown => &block.feedforward.down,
                WeightKind::FeedForwardDownBias => &block.feedforward.down_bias,
                _ => return Err(invalid(format!("vision weight role {role:?} is invalid"))),
            }
        }
        (WeightScope::Vision, kind) => match kind {
            WeightKind::PatchBias => &resident.patch_bias,
            WeightKind::PositionEmbedding => &resident.position_embedding,
            WeightKind::NormWeight => &resident.output_norm.weight,
            WeightKind::NormBias => &resident.output_norm.bias,
            WeightKind::MergerHidden => &resident.merger.hidden,
            WeightKind::MergerHiddenBias => &resident.merger.hidden_bias,
            WeightKind::MergerOutput => &resident.merger.output,
            WeightKind::MergerOutputBias => &resident.merger.output_bias,
            _ => return Err(invalid(format!("vision weight role {role:?} is invalid"))),
        },
        _ => return Err(invalid(format!("vision weight role {role:?} is invalid"))),
    };
    Ok(weight)
}

impl VisionProgram for NativeVisionProgram {
    type Submission =
        ReadySubmission<VisionLaunchCore, NativeGraphWorkspaceLease, GraphOutputTensor>;
    fn submit(
        &mut self,
        mut launch: ValidatedVisionLaunch,
    ) -> Result<Self::Submission, (SubmitError, ValidatedVisionLaunch)> {
        let result = {
            let (core, workspace, output) = launch.execution_parts_mut();
            self.execute_graph(core, workspace, output)
        };
        let features = match result {
            Ok(features) => features,
            Err(error) => return Err((error, launch)),
        };
        let (core, workspace, _) = launch.into_submission_parts();
        Ok(ReadySubmission::new(core, workspace, features))
    }
}
