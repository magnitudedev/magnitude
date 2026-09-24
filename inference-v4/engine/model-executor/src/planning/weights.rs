use super::{ArtifactComponent, ArtifactComponentKind, ComponentSelection, ProgramPlan};
use crate::error::PlanError;
use magnitude_artifacts::{
    gguf::{Encoding, TensorDescriptor},
    PackageManifest,
};
use magnitude_model_contracts::{
    ActivationDType, AttentionWeights, BlockWeights, DenseFeedForwardWeights, FeedForwardWeights,
    LayerNormWeights, MixerWeights, ModelDefinition, RecurrentWeights, RoutedFeedForwardWeights,
    VisionDescription, WeightDescriptor, WeightKind, WeightRole, WeightScope,
};
use crate::ExecutionPath;
use seismic::{BackendName, DType, Element, Layout};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightPlan {
    pub role: WeightRole,
    pub component: ArtifactComponent,
    pub source: Element,
    pub resident: Element,
    pub shape: Vec<u64>,
    pub descriptor: WeightDescriptor,
    /// Startup upload charge for the admitted artifact encoding.
    pub source_bytes: u64,
    pub resident_bytes: u64,
}

/// Physical resident storage identity. Multiple semantic roles may name one
/// tensor, while distinct resident representations require distinct storage.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WeightStorageIdentity {
    pub component: ArtifactComponent,
    pub tensor_name: String,
    pub resident: Element,
}

impl WeightPlan {
    pub fn storage_identity(&self) -> WeightStorageIdentity {
        WeightStorageIdentity {
            component: self.component,
            tensor_name: self.descriptor.name.clone(),
            resident: self.resident,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EmbeddingBinding {
    pub table: Element,
    pub activation: Element,
}

/// The dimensions attention kernels are specialized to: the hidden width,
/// kv heads, query heads per kv head, rotated pairs and head width.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttentionShape {
    pub hidden: u64,
    pub kv_heads: u64,
    pub group: u64,
    pub rotary_pairs: u64,
    pub width: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttentionBinding {
    pub shape: AttentionShape,
    pub norm: Element,
    pub query_gate: Element,
    pub key: Element,
    pub value: Element,
    pub output: Element,
    pub activation: Element,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RecurrentBinding {
    pub key_heads: u64,
    pub value_heads: u64,
    pub width: u64,
    pub convolution_width: u64,
    pub norm: Element,
    pub qkv: Element,
    pub gate: Element,
    pub alpha: Element,
    pub beta: Element,
    pub recurrent_norm: Element,
    pub output: Element,
    pub activation: Element,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DenseBinding {
    pub norm: Element,
    pub gate: Element,
    pub up: Element,
    pub down: Element,
    pub activation: Element,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RoutedBinding {
    pub hidden: u64,
    pub experts: u64,
    pub selected: u64,
    pub features: u64,
    pub shared: u64,
    pub norm: Element,
    pub router: Element,
    pub expert_gate: Element,
    pub expert_up: Element,
    pub expert_down: Element,
    pub shared_gate: Element,
    pub shared_up: Element,
    pub shared_down: Element,
    pub activation: Element,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReadoutBinding {
    pub norm: Element,
    pub weight: Element,
    pub activation: Element,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FeaturesBinding {
    pub norm: Element,
    pub activation: Element,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HeadBinding {
    pub attention_shape: AttentionShape,
    pub embedding_table: Element,
    pub embedding_norm: Element,
    pub hidden_norm: Element,
    pub combine: Element,
    pub input_norm: Element,
    pub query_gate: Element,
    pub key: Element,
    pub value: Element,
    pub attention_output: Element,
    pub feedforward_norm: Element,
    pub gate: Element,
    pub up: Element,
    pub down: Element,
    pub output_norm: Element,
    pub projection: Element,
    pub activation: Element,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VisionPatchBinding {
    pub temporal_weight_0: Element,
    pub temporal_weight_1: Element,
    pub bias: Element,
    pub position: Element,
    pub activation: Element,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VisionBlockBinding {
    pub input_norm_weight: Element,
    pub input_norm_bias: Element,
    pub qkv_weight: Element,
    pub qkv_bias: Element,
    pub attention_output: Element,
    pub attention_output_bias: Element,
    pub feedforward_norm_weight: Element,
    pub feedforward_norm_bias: Element,
    pub up: Element,
    pub up_bias: Element,
    pub down: Element,
    pub down_bias: Element,
    pub activation: Element,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VisionMergerBinding {
    pub output_norm_weight: Element,
    pub output_norm_bias: Element,
    pub hidden: Element,
    pub hidden_bias: Element,
    pub output: Element,
    pub output_bias: Element,
    pub activation: Element,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelLoadPlan {
    pub(super) target: Vec<WeightPlan>,
    pub(super) head: Option<Vec<WeightPlan>>,
    pub(super) vision: Option<Vec<WeightPlan>>,
}

pub(super) fn weight_bytes_by_component(load: &ModelLoadPlan) -> Result<[u64; 3], String> {
    let mut seen = HashMap::new();
    let mut bytes = [0u64; 3];
    for (index, weights) in [
        load.target.as_slice(),
        load.head.as_deref().unwrap_or_default(),
        load.vision.as_deref().unwrap_or_default(),
    ]
    .into_iter()
    .enumerate()
    {
        for weight in weights {
            match seen.entry(weight.storage_identity()) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert((
                        weight.source,
                        weight.shape.clone(),
                        weight.source_bytes,
                        weight.resident_bytes,
                    ));
                    bytes[index] = bytes[index]
                        .checked_add(weight.resident_bytes)
                        .ok_or("resident weight byte count overflow")?;
                }
                std::collections::hash_map::Entry::Occupied(entry)
                    if entry.get()
                        != &(
                            weight.source,
                            weight.shape.clone(),
                            weight.source_bytes,
                            weight.resident_bytes,
                        ) =>
                {
                    return Err("tied weight roles disagree on their physical storage".into());
                }
                std::collections::hash_map::Entry::Occupied(_) => {}
            }
        }
    }
    Ok(bytes)
}

impl ModelLoadPlan {
    pub fn target(&self) -> &[WeightPlan] {
        &self.target
    }

    pub fn head(&self) -> Option<&[WeightPlan]> {
        self.head.as_deref()
    }

    pub fn vision(&self) -> Option<&[WeightPlan]> {
        self.vision.as_deref()
    }

    /// Resolve the ordered checked-entry topology from the same weight facts
    /// used for import and residency.
    pub(crate) fn program_plan(
        &self,
        definition: &ModelDefinition,
    ) -> Result<ProgramPlan, PlanError> {
        super::programs::derive_program_plan(
            definition,
            &self.target,
            self.head.as_deref(),
            self.vision.as_deref(),
        )
    }

    /// Plan every selected weight. Packed weights become resident in
    /// `layout` (`resident_layout` of the engine's execution path and
    /// device backend).
    pub fn derive(
        manifest: &PackageManifest,
        definition: &ModelDefinition,
        selection: ComponentSelection,
        layout: Layout,
    ) -> Result<Self, String> {
        definition.validate().map_err(|error| error.to_string())?;
        if manifest.identity != definition.artifact_identity {
            return Err("load plan package identity mismatch".into());
        }
        let target_component = ArtifactComponent {
            kind: ArtifactComponentKind::Target,
            identity: manifest.target.identity,
        };
        let target_form = ResidentForm {
            activation: activation_dtype(definition.geometry.activation_dtype),
            layout,
        };
        let mut target = Vec::new();
        push_weight(
            &mut target,
            &manifest.target.tensors,
            target_component,
            WeightScope::Target,
            WeightKind::Embedding,
            &definition.embedding,
            target_form,
        )?;
        for (index, block) in definition.blocks.iter().enumerate() {
            let index = u32::try_from(index).map_err(|_| "target block index exceeds u32")?;
            append_block(
                &mut target,
                &manifest.target.tensors,
                target_component,
                WeightScope::TargetBlock(index),
                block,
                target_form,
            )?;
        }
        push_weight(
            &mut target,
            &manifest.target.tensors,
            target_component,
            WeightScope::Target,
            WeightKind::OutputNorm,
            &definition.output_norm,
            target_form,
        )?;
        push_weight(
            &mut target,
            &manifest.target.tensors,
            target_component,
            WeightScope::Target,
            WeightKind::Output,
            &definition.output,
            target_form,
        )?;

        if selection.head && definition.head.is_none() {
            return Err("head execution was selected without a head definition".into());
        }
        if manifest.projector.is_some() != definition.vision.is_some() {
            return Err("projector component and vision definition disagree".into());
        }
        if selection.vision && definition.vision.is_none() {
            return Err("vision execution was selected without a vision definition".into());
        }
        let head = selection
            .head
            .then_some(definition.head.as_ref())
            .flatten()
            .map(|head| {
                let mut plans = Vec::new();
                for (index, block) in head.blocks.iter().enumerate() {
                    let scope = WeightScope::HeadBlock(
                        u32::try_from(index).map_err(|_| "head block index exceeds u32")?,
                    );
                    for (kind, descriptor) in [
                        (WeightKind::HeadEmbeddingNorm, &block.embedding_norm),
                        (WeightKind::HeadHiddenNorm, &block.hidden_norm),
                        (WeightKind::HeadCombine, &block.combine),
                        (WeightKind::InputNorm, &block.input_norm),
                    ] {
                        push_weight(
                            &mut plans,
                            &manifest.target.tensors,
                            target_component,
                            scope,
                            kind,
                            descriptor,
                            target_form,
                        )?;
                    }
                    append_attention(
                        &mut plans,
                        &manifest.target.tensors,
                        target_component,
                        scope,
                        &block.attention,
                        target_form,
                    )?;
                    push_weight(
                        &mut plans,
                        &manifest.target.tensors,
                        target_component,
                        scope,
                        WeightKind::FeedForwardNorm,
                        &block.feedforward_norm,
                        target_form,
                    )?;
                    append_dense(
                        &mut plans,
                        &manifest.target.tensors,
                        target_component,
                        scope,
                        &block.feedforward,
                        target_form,
                    )?;
                    push_weight(
                        &mut plans,
                        &manifest.target.tensors,
                        target_component,
                        scope,
                        WeightKind::OutputNorm,
                        &block.output_norm,
                        target_form,
                    )?;
                }
                Ok::<_, String>(plans)
            })
            .transpose()?;

        let vision = selection
            .vision
            .then_some(definition.vision.as_ref())
            .flatten()
            .map(|vision| {
                let component_manifest = manifest
                    .projector
                    .as_ref()
                    .ok_or("vision definition requires a projector component")?;
                let component = ArtifactComponent {
                    kind: ArtifactComponentKind::Projector,
                    identity: component_manifest.identity,
                };
                plan_vision(component_manifest.tensors.as_slice(), component, vision, layout)
            })
            .transpose()?;
        validate_unique_roles(
            target
                .iter()
                .chain(head.as_deref().unwrap_or_default())
                .chain(vision.as_deref().unwrap_or_default()),
        )?;

        let load = Self {
            target,
            head,
            vision,
        };
        load.program_plan(definition)
            .map_err(|error| error.to_string())?;
        Ok(load)
    }

    pub fn weights(&self) -> impl Iterator<Item = &WeightPlan> {
        self.target
            .iter()
            .chain(self.head.iter().flatten())
            .chain(self.vision.iter().flatten())
    }
}

pub(super) fn validate_unique_roles<'a>(
    plans: impl Iterator<Item = &'a WeightPlan>,
) -> Result<(), String> {
    let mut roles = HashSet::new();
    for plan in plans {
        if !roles.insert((plan.component, plan.role)) {
            return Err(format!(
                "load plan contains duplicate semantic role {:?}/{:?}",
                plan.component, plan.role
            ));
        }
    }
    Ok(())
}

pub(super) fn activation_dtype(dtype: ActivationDType) -> DType {
    match dtype {
        ActivationDType::F16 => DType::F16,
        ActivationDType::BF16 => DType::BF16,
    }
}

fn push_weight(
    out: &mut Vec<WeightPlan>,
    inventory: &[TensorDescriptor],
    component: ArtifactComponent,
    scope: WeightScope,
    kind: WeightKind,
    descriptor: &WeightDescriptor,
    form: ResidentForm,
) -> Result<(), String> {
    let stored = inventory
        .iter()
        .find(|tensor| tensor.name == descriptor.name)
        .ok_or_else(|| {
            format!(
                "planned weight {:?} is absent from its component",
                descriptor.name
            )
        })?;
    if stored.shape != descriptor.shape {
        return Err(format!(
            "planned weight {:?} changed shape",
            descriptor.name
        ));
    }
    let role = WeightRole { scope, kind };
    let dense_resident = resident_dtype(role, form.activation);
    if is_fixed_dense_role(kind)
        && !matches!(
            stored.encoding,
            Encoding::F32 | Encoding::F16 | Encoding::BF16
        )
    {
        return Err(format!(
            "fixed-f32 weight {scope:?}/{kind:?} cannot use packed source encoding {:?}",
            stored.encoding
        ));
    }
    let source = source_element(stored.encoding)
        .ok_or_else(|| format!("unsupported source encoding {:?}", stored.encoding))?;
    let resident = resident_element(stored.encoding, dense_resident, form.layout).ok_or_else(|| {
        format!(
            "{:?} has no resident form in the `{}` layout",
            stored.encoding,
            form.layout.as_str()
        )
    })?;
    let source_bytes = representation_bytes(source, &stored.shape)?;
    if source_bytes != stored.nbytes {
        return Err(format!(
            "weight {:?} ({role:?}) stores {} bytes, but its {:?} shape {:?} is {source_bytes}",
            descriptor.name, stored.nbytes, stored.encoding, stored.shape
        ));
    }
    let resident_bytes = representation_bytes(resident, &stored.shape)?;
    out.push(WeightPlan {
        role,
        component,
        source,
        resident,
        shape: descriptor.shape.clone(),
        descriptor: descriptor.clone(),
        source_bytes: stored.nbytes,
        resident_bytes,
    });
    Ok(())
}

/// The resident representation is part of the semantic kernel ABI, not a
/// blanket model-wide preference. These roles are consumed by fixed-f32
/// kernel arguments; every other weight remains representation-generic and
/// follows the component activation dtype (or its admitted packed format).
pub(super) fn resident_dtype(role: WeightRole, activation: DType) -> DType {
    if is_fixed_dense_role(role.kind) {
        DType::F32
    } else {
        activation
    }
}

fn is_fixed_dense_role(kind: WeightKind) -> bool {
    matches!(
        kind,
        WeightKind::QueryNorm
            | WeightKind::KeyNorm
            | WeightKind::RecurrentConvolution
            | WeightKind::RecurrentDecay
            | WeightKind::RecurrentTimeBias
            | WeightKind::SharedRouter
    )
}

fn append_block(
    out: &mut Vec<WeightPlan>,
    inventory: &[TensorDescriptor],
    component: ArtifactComponent,
    scope: WeightScope,
    block: &BlockWeights,
    form: ResidentForm,
) -> Result<(), String> {
    push_weight(
        out,
        inventory,
        component,
        scope,
        WeightKind::InputNorm,
        &block.input_norm,
        form,
    )?;
    match &block.mixer {
        MixerWeights::Attention(weights) => {
            append_attention(out, inventory, component, scope, weights, form)?
        }
        MixerWeights::Recurrent(weights) => {
            append_recurrent(out, inventory, component, scope, weights, form)?
        }
    }
    push_weight(
        out,
        inventory,
        component,
        scope,
        WeightKind::FeedForwardNorm,
        &block.feedforward_norm,
        form,
    )?;
    match &block.feedforward {
        FeedForwardWeights::Dense(weights) => {
            append_dense(out, inventory, component, scope, weights, form)
        }
        FeedForwardWeights::Routed(weights) => {
            append_routed(out, inventory, component, scope, weights, form)
        }
    }
}

fn append_attention(
    out: &mut Vec<WeightPlan>,
    inventory: &[TensorDescriptor],
    component: ArtifactComponent,
    scope: WeightScope,
    weights: &AttentionWeights,
    form: ResidentForm,
) -> Result<(), String> {
    for (kind, descriptor) in [
        (WeightKind::QueryGate, &weights.query_gate),
        (WeightKind::Key, &weights.key),
        (WeightKind::Value, &weights.value),
        (WeightKind::QueryNorm, &weights.query_norm),
        (WeightKind::KeyNorm, &weights.key_norm),
        (WeightKind::AttentionOutput, &weights.output),
    ] {
        push_weight(out, inventory, component, scope, kind, descriptor, form)?;
    }
    Ok(())
}

fn append_recurrent(
    out: &mut Vec<WeightPlan>,
    inventory: &[TensorDescriptor],
    component: ArtifactComponent,
    scope: WeightScope,
    weights: &RecurrentWeights,
    form: ResidentForm,
) -> Result<(), String> {
    for (kind, descriptor) in [
        (WeightKind::RecurrentQueryKeyValue, &weights.query_key_value),
        (WeightKind::RecurrentGate, &weights.gate),
        (WeightKind::RecurrentAlpha, &weights.alpha),
        (WeightKind::RecurrentBeta, &weights.beta),
        (WeightKind::RecurrentConvolution, &weights.convolution),
        (WeightKind::RecurrentDecay, &weights.decay),
        (WeightKind::RecurrentTimeBias, &weights.time_bias),
        (WeightKind::RecurrentNorm, &weights.norm),
        (WeightKind::RecurrentOutput, &weights.output),
    ] {
        push_weight(out, inventory, component, scope, kind, descriptor, form)?;
    }
    Ok(())
}

fn append_dense(
    out: &mut Vec<WeightPlan>,
    inventory: &[TensorDescriptor],
    component: ArtifactComponent,
    scope: WeightScope,
    weights: &DenseFeedForwardWeights,
    form: ResidentForm,
) -> Result<(), String> {
    for (kind, descriptor) in [
        (WeightKind::DenseGate, &weights.gate),
        (WeightKind::DenseUp, &weights.up),
        (WeightKind::DenseDown, &weights.down),
    ] {
        push_weight(out, inventory, component, scope, kind, descriptor, form)?;
    }
    Ok(())
}

fn append_routed(
    out: &mut Vec<WeightPlan>,
    inventory: &[TensorDescriptor],
    component: ArtifactComponent,
    scope: WeightScope,
    weights: &RoutedFeedForwardWeights,
    form: ResidentForm,
) -> Result<(), String> {
    for (kind, descriptor) in [
        (WeightKind::Router, &weights.router),
        (WeightKind::SharedRouter, &weights.shared_router),
        (WeightKind::ExpertGate, &weights.expert_gate),
        (WeightKind::ExpertUp, &weights.expert_up),
        (WeightKind::ExpertDown, &weights.expert_down),
        (WeightKind::SharedGate, &weights.shared_gate),
        (WeightKind::SharedUp, &weights.shared_up),
        (WeightKind::SharedDown, &weights.shared_down),
    ] {
        push_weight(out, inventory, component, scope, kind, descriptor, form)?;
    }
    Ok(())
}

fn push_norm(
    out: &mut Vec<WeightPlan>,
    inventory: &[TensorDescriptor],
    component: ArtifactComponent,
    scope: WeightScope,
    weight_kind: WeightKind,
    bias_kind: WeightKind,
    norm: &LayerNormWeights,
    form: ResidentForm,
) -> Result<(), String> {
    push_weight(
        out,
        inventory,
        component,
        scope,
        weight_kind,
        &norm.weight,
        form,
    )?;
    push_weight(
        out, inventory, component, scope, bias_kind, &norm.bias, form,
    )
}

fn plan_vision(
    inventory: &[TensorDescriptor],
    component: ArtifactComponent,
    vision: &VisionDescription,
    layout: Layout,
) -> Result<Vec<WeightPlan>, String> {
    let form = ResidentForm {
        activation: activation_dtype(vision.geometry.activation_dtype),
        layout,
    };
    let mut out = Vec::new();
    for (index, descriptor) in vision.patch_embeddings.iter().enumerate() {
        let scope = WeightScope::VisionPatch(
            u32::try_from(index).map_err(|_| "vision patch index exceeds u32")?,
        );
        push_weight(
            &mut out,
            inventory,
            component,
            scope,
            WeightKind::PatchEmbedding,
            descriptor,
            form,
        )?;
    }
    push_weight(
        &mut out,
        inventory,
        component,
        WeightScope::Vision,
        WeightKind::PatchBias,
        &vision.patch_bias,
        form,
    )?;
    push_weight(
        &mut out,
        inventory,
        component,
        WeightScope::Vision,
        WeightKind::PositionEmbedding,
        &vision.position_embedding,
        form,
    )?;
    for (index, block) in vision.blocks.iter().enumerate() {
        let scope = WeightScope::VisionBlock(
            u32::try_from(index).map_err(|_| "vision block index exceeds u32")?,
        );
        push_norm(
            &mut out,
            inventory,
            component,
            scope,
            WeightKind::InputNormWeight,
            WeightKind::InputNormBias,
            &block.input_norm,
            form,
        )?;
        push_weight(
            &mut out,
            inventory,
            component,
            scope,
            WeightKind::FusedQkvWeight,
            &block.attention.qkv.weight,
            form,
        )?;
        push_weight(
            &mut out,
            inventory,
            component,
            scope,
            WeightKind::FusedQkvBias,
            &block.attention.qkv.bias,
            form,
        )?;
        push_weight(
            &mut out,
            inventory,
            component,
            scope,
            WeightKind::AttentionOutput,
            &block.attention.output,
            form,
        )?;
        push_weight(
            &mut out,
            inventory,
            component,
            scope,
            WeightKind::AttentionOutputBias,
            &block.attention.output_bias,
            form,
        )?;
        push_norm(
            &mut out,
            inventory,
            component,
            scope,
            WeightKind::FeedForwardNormWeight,
            WeightKind::FeedForwardNormBias,
            &block.feedforward_norm,
            form,
        )?;
        for (kind, descriptor) in [
            (WeightKind::DenseUp, &block.feedforward.up),
            (WeightKind::FeedForwardUpBias, &block.feedforward.up_bias),
            (WeightKind::DenseDown, &block.feedforward.down),
            (
                WeightKind::FeedForwardDownBias,
                &block.feedforward.down_bias,
            ),
        ] {
            push_weight(
                &mut out, inventory, component, scope, kind, descriptor, form,
            )?;
        }
    }
    push_norm(
        &mut out,
        inventory,
        component,
        WeightScope::Vision,
        WeightKind::NormWeight,
        WeightKind::NormBias,
        &vision.output_norm,
        form,
    )?;
    for (kind, descriptor) in [
        (WeightKind::MergerHidden, &vision.merger.hidden),
        (WeightKind::MergerHiddenBias, &vision.merger.hidden_bias),
        (WeightKind::MergerOutput, &vision.merger.output),
        (WeightKind::MergerOutputBias, &vision.merger.output_bias),
    ] {
        push_weight(
            &mut out,
            inventory,
            component,
            WeightScope::Vision,
            kind,
            descriptor,
            form,
        )?;
    }
    Ok(out)
}

pub(super) fn planned_element(
    plans: &[WeightPlan],
    scope: WeightScope,
    kind: WeightKind,
) -> Result<Element, String> {
    plans
        .iter()
        .find(|plan| plan.role == WeightRole { scope, kind })
        .map(|plan| plan.resident)
        .ok_or_else(|| format!("load plan is missing {scope:?}/{kind:?}"))
}

/// How the weights of one component become resident: the activation dtype
/// dense weights follow, and the layout packed weights are stored in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResidentForm {
    activation: DType,
    layout: Layout,
}

/// The layout resident packed weights use for an execution path on a device
/// backend (spec S1/E7). Native kernels read the backend's execution layout:
/// Metal and Vulkan `rows16`, CUDA `mma16`; native CPU kernels and every planned
/// (compiled) kernel read the `packet` layout.
pub fn resident_layout(path: ExecutionPath, backend: BackendName) -> Layout {
    match (path, backend) {
        (ExecutionPath::Native, BackendName::Metal | BackendName::Vulkan) => Layout::Rows16,
        (ExecutionPath::Native, BackendName::Cuda) => Layout::Mma16,
        (ExecutionPath::Native, BackendName::Cpu) | (ExecutionPath::Planned, _) => Layout::Packet,
    }
}

/// The external storage of an artifact encoding.
pub fn source_element(encoding: Encoding) -> Option<Element> {
    match encoding {
        Encoding::F32 => Some(Element::dense(DType::F32)),
        Encoding::F16 => Some(Element::dense(DType::F16)),
        Encoding::BF16 => Some(Element::dense(DType::BF16)),
        Encoding::Q8_0 => Element::named("gguf_q8_0"),
        Encoding::Q4K => Element::named("gguf_q4_k"),
        Encoding::Q5K => Element::named("gguf_q5_k"),
        Encoding::Q6K => Element::named("gguf_q6_k"),
        Encoding::Iq4Xs => Element::named("gguf_iq4_xs"),
        _ => None,
    }
}

/// The one map from an artifact encoding to resident storage: dense
/// encodings become `dense`; packed encodings become their representation
/// (from the source format) in `layout` (from the device backend).
pub fn resident_element(encoding: Encoding, dense: DType, layout: Layout) -> Option<Element> {
    let representation = match encoding {
        Encoding::F32 | Encoding::F16 | Encoding::BF16 => return Some(Element::dense(dense)),
        Encoding::Q8_0 => "q8g32s",
        Encoding::Q4K => "q4k",
        Encoding::Q5K => "q5k",
        Encoding::Q6K => "q6k",
        Encoding::Iq4Xs => "iq4g32",
        _ => return None,
    };
    Element::stored(representation, layout)
}

fn representation_bytes(element: Element, shape: &[u64]) -> Result<u64, String> {
    element
        .canonical_byte_len(shape)
        .map_err(|error| format!("{} shape {shape:?}: {error}", element.name()))
}

#[cfg(test)]
mod representation_byte_tests {
    use super::*;

    #[test]
    fn packed_residency_charges_each_row_and_packet_alignment() {
        let q4 = Element::named("q4k").unwrap();
        // Two half-packet rows occupy two packets, even though their total
        // logical element count is only one full packet.
        assert_eq!(representation_bytes(q4, &[2, 128]).unwrap(), 288);

        let q8 = Element::named("q8g32s").unwrap();
        // The resident packet has aligned planes, unlike the 34-byte GGUF
        // source packet.
        assert_eq!(representation_bytes(q8, &[2, 32]).unwrap(), 72);

        let q6 = Element::named("q6k").unwrap();
        assert_eq!(representation_bytes(q6, &[1, 256]).unwrap(), 212);
        assert_eq!(representation_bytes(Element::f16(), &[2, 3]).unwrap(), 12);
    }

    #[test]
    fn row_layouts_pad_rows_to_sixteen_bytes_and_mma16_rows_to_tiles() {
        // Qwen3.5 K = 2560, q4k: codes 1280 | scales 120 -> 128 | supers 40 -> 48.
        let rows16 = Element::stored("q4k", Layout::Rows16).unwrap();
        assert_eq!(representation_bytes(rows16, &[3, 2560]).unwrap(), 3 * 1456);
        let mma16 = Element::stored("q4k", Layout::Mma16).unwrap();
        assert_eq!(representation_bytes(mma16, &[17, 2560]).unwrap(), 32 * 1456);
        // A q6k row of one group: codes 128 | 64 | scales 16 | supers 2 -> 16.
        let q6 = Element::stored("q6k", Layout::Rows16).unwrap();
        assert_eq!(representation_bytes(q6, &[1, 256]).unwrap(), 224);
    }

    #[test]
    fn one_map_chooses_representation_from_format_and_layout_from_backend() {
        for (path, backend, layout) in [
            (ExecutionPath::Native, BackendName::Metal, Layout::Rows16),
            (ExecutionPath::Native, BackendName::Vulkan, Layout::Rows16),
            (ExecutionPath::Native, BackendName::Cuda, Layout::Mma16),
            (ExecutionPath::Native, BackendName::Cpu, Layout::Packet),
            (ExecutionPath::Planned, BackendName::Metal, Layout::Packet),
        ] {
            assert_eq!(resident_layout(path, backend), layout);
        }
        for (encoding, representation) in [
            (Encoding::Q8_0, "q8g32s"),
            (Encoding::Q4K, "q4k"),
            (Encoding::Q5K, "q5k"),
            (Encoding::Q6K, "q6k"),
            (Encoding::Iq4Xs, "iq4g32"),
        ] {
            for layout in Layout::ALL {
                let element = resident_element(encoding, DType::BF16, layout).unwrap();
                assert_eq!((element.representation(), element.layout()), (representation, layout));
            }
        }
        assert_eq!(
            resident_element(Encoding::F16, DType::BF16, Layout::Rows16),
            Some(Element::bf16())
        );
        assert_eq!(resident_element(Encoding::Q3K, DType::BF16, Layout::Rows16), None);
    }
}
