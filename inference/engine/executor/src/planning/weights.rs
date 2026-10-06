use super::{
    ArtifactComponent, ArtifactComponentKind, ComponentSelection, FeedForwardProgramSlot,
    ProgramPlan,
};
use crate::error::PlanError;
use crate::{operators, ExecutionPath};
use magnitude_artifacts::{
    gguf::{Encoding, TensorDescriptor},
    ArtifactIdentity, PackageHeaders, PackageIdentity, PackageManifest,
};
use magnitude_family_contracts::{
    ActivationDType, ImportTransform, ModelDefinition, ProgressivePlane, VisionDescription,
    WeightDescriptor, WeightKind, WeightRole, WeightScope,
};
use magnitude_state::KvCodec;
use seismic::{BackendName, DType, Element, Layout};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightPlan {
    pub role: WeightRole,
    pub component: ArtifactComponent,
    /// The stored representation of the artifact tensor.
    pub source: Element,
    /// The representation the importer uploads: `source`, or F32 for a
    /// packed weight the host dequantizes exactly (`import_transforms::
    /// dequantize`) because its kernels read dense weights only.
    pub upload: Element,
    pub resident: Element,
    pub shape: Vec<u64>,
    pub descriptor: WeightDescriptor,
    /// Startup upload charge: the logical shape in `upload`.
    pub source_bytes: u64,
    /// The resident representation's bytes, without `scale`.
    pub resident_bytes: u64,
    /// The weight's resident second-level scale (`ImportTransform::
    /// ScaleByTensor`), when it has one.
    pub scale: Option<WeightScalePlan>,
}

impl WeightPlan {
    /// Whether the importer uploads host-prepared bytes (transformed or
    /// dequantized) rather than the stored range as is. A second-level
    /// scale leaves the weight's bytes as stored.
    pub fn host_prepared(&self) -> bool {
        self.descriptor
            .transforms
            .iter()
            .any(|transform| !matches!(transform, ImportTransform::ScaleByTensor { .. }))
            || self.upload != self.source
    }

    /// Whether the host places the weight's resident bytes itself (a
    /// progressive plane): the device holds them as uploaded, with no import
    /// entry.
    pub fn placed_on_host(&self) -> bool {
        matches!(self.descriptor.transforms[..], [ImportTransform::Progressive(_)])
    }

    /// Every resident byte of the weight: its representation and its scale.
    pub fn storage_bytes(&self) -> Result<u64, String> {
        self.resident_bytes
            .checked_add(self.scale.as_ref().map_or(0, |scale| scale.resident_bytes))
            .ok_or_else(|| "resident weight byte count overflows".into())
    }

    /// The extent of the weight's accumulator-scale port: 0 without a
    /// second-level scale, else its resident scale's (1 for a matrix, the
    /// stack's matrix count for a stacked weight).
    pub fn scale_extent(&self) -> u64 {
        self.scale.as_ref().map_or(0, |scale| scale.extent)
    }
}

/// A packed weight's second-level scale (NVFP4's per-tensor or per-matrix
/// F32 `.scale`): a stored tensor of its own, `[1]` or `[stack]`, resident
/// beside the weight's representation as one F32 value per matrix. Entries
/// multiply their F32 accumulator by it through their accumulator-scale
/// port; an entry without one refuses the weight at plan time.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WeightScalePlan {
    /// The stored scale tensor.
    pub tensor: String,
    /// Its stored extent: 1, or the weight's matrix count.
    pub stored: u64,
    /// Its resident extent: 1 for a matrix, the matrix count for a stacked
    /// weight (a stored `[1]` is repeated per matrix).
    pub extent: u64,
    /// `extent` F32 values.
    pub resident_bytes: u64,
}

/// A projection weight's planned form at an entry with an accumulator-scale
/// port: its resident element and the port's extent (0 without a scale).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScalableWeight {
    pub element: Element,
    pub scale: u64,
}

/// Physical resident storage identity. Multiple semantic roles may name one
/// tensor, while distinct resident representations, and distinct import
/// transforms of one stored tensor, require distinct storage.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WeightStorageIdentity {
    pub component: ArtifactComponent,
    pub tensor_name: String,
    pub transforms: Vec<ImportTransform>,
    pub resident: Element,
}

impl WeightPlan {
    pub fn storage_identity(&self) -> WeightStorageIdentity {
        WeightStorageIdentity {
            component: self.component,
            tensor_name: self.descriptor.name.clone(),
            transforms: self.descriptor.transforms.clone(),
            resident: self.resident,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EmbeddingBinding {
    pub table: Element,
    pub activation: Element,
}

/// The static axes the attention entries are specialized to. Optional
/// parts of the operator are axes of extent 0 or 1 (`attention.seismic`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttentionShape {
    /// `D`: the hidden width.
    pub hidden: u64,
    /// `KV`: key/value heads.
    pub kv_heads: u64,
    /// `G`: query heads per key/value head.
    pub group: u64,
    /// `P`: rotated pairs of a head.
    pub rotary_pairs: u64,
    /// `W = 2P + S`: the head width.
    pub width: u64,
    /// `I` ∈ {0, W}: gate columns after each query head's W columns.
    pub interleaved_gate: u64,
    /// `U` ∈ {0, 1, W}: separate gate values per query head.
    pub separate_gate: u64,
    /// `F` ∈ {0, 1}: whether the layer projects and appends its own keys and
    /// values.
    pub fresh: u64,
    /// `N` ∈ {0, 1}: whether queries and keys are RMS-normalized per head.
    pub head_norm: u64,
    /// `NV` ∈ {0, 1}: whether values are RMS-normalized per head.
    pub value_norm: u64,
    /// Whether values have their own projection (else they are the raw key).
    pub projected_value: bool,
}

impl AttentionShape {
    pub fn heads(&self) -> u64 {
        self.kv_heads * self.group
    }

    /// `Q`: query (and interleaved gate) projection rows.
    pub fn query_rows(&self) -> u64 {
        self.heads() * (self.width + self.interleaved_gate)
    }

    /// `GR`: separate gate projection rows.
    pub fn gate_rows(&self) -> u64 {
        self.heads() * self.separate_gate
    }

    /// `K`: key projection rows.
    pub fn key_rows(&self) -> u64 {
        self.fresh * self.kv_heads * self.width
    }

    /// `V`: value projection rows.
    pub fn value_rows(&self) -> u64 {
        if self.projected_value {
            self.key_rows()
        } else {
            0
        }
    }

    /// Unrotated columns of a head.
    pub fn static_width(&self) -> u64 {
        self.width - 2 * self.rotary_pairs
    }

    /// The projection entry's dimensions at `rows` rows.
    pub fn project_dimensions(&self, rows: u64) -> [(&'static str, u64); 6] {
        [
            ("M", rows),
            ("D", self.hidden),
            ("Q", self.query_rows()),
            ("GR", self.gate_rows()),
            ("K", self.key_rows()),
            ("V", self.value_rows()),
        ]
    }

    /// The static axes of the fused attention entries.
    pub fn mix_statics(&self) -> [(&'static str, u64); 9] {
        [
            ("KV", self.kv_heads),
            ("G", self.group),
            ("P", self.rotary_pairs),
            ("S", self.static_width()),
            ("I", self.interleaved_gate),
            ("U", self.separate_gate),
            ("F", self.fresh),
            ("N", self.head_norm),
            ("NV", self.value_norm),
        ]
    }

    /// The fused entries' dimensions for `rows` rows over `history_rows`
    /// history rows read in `segments` visible spans.
    pub fn mix_dimensions(
        &self,
        rows: u64,
        history_rows: u64,
        segments: u64,
    ) -> [(&'static str, u64); 12] {
        let [kv, g, p, s, i, u, f, n, nv] = self.mix_statics();
        [
            ("M", rows),
            ("T", history_rows),
            kv,
            g,
            p,
            s,
            i,
            u,
            f,
            n,
            nv,
            ("R", segments),
        ]
    }
}

/// An attention operator's kernel binding. Weights a form lacks (a separate
/// gate, own keys or values) are empty segments of the projection; they
/// bind a zero-row view of the query weight and carry its element.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttentionBinding {
    pub shape: AttentionShape,
    pub norm: Element,
    pub query: Element,
    pub gate: Element,
    pub key: Element,
    pub value: Element,
    pub output: Element,
    pub activation: Element,
    /// How the block's history planes encode keys and values.
    pub history: KvCodec,
    pub tail: SublayerTail,
}

/// How a sublayer's output projection joins the residual stream (the
/// contract's `OutputForm`): added by the operator's own output entry, or
/// projected to F32 and added through a row op that normalizes it first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SublayerTail {
    Residual,
    /// `(residual + RMS(projection)·norm)·scale`, the norm weight's element;
    /// `scaled` when the scale is the layer's output scale (else 1).
    PostNorm { norm: Element, scaled: bool },
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
    /// The expansion width (`F`), a static of the specializations: layers
    /// of different widths (Gemma's double-wide layers) need their own.
    pub features: u64,
    pub norm: Element,
    pub gate: Element,
    pub up: Element,
    pub down: Element,
    pub activation: Element,
    pub tail: SublayerTail,
    pub scales: DenseScales,
}

/// The extents of a dense feed-forward's per-tensor accumulator-scale ports
/// (`dense_expand`/`dense_up` gate and up, `dense_output` or the post-norm
/// tail's `project_rows` down): 1 where the weight has a resident
/// second-level scale, else 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DenseScales {
    pub gate: u64,
    pub up: u64,
    pub down: u64,
}

/// A dense branch beside a general routed branch
/// (`operators::DenseBesideRouted`): the dense branch's expansion and its
/// down projection into F32, the routed branch summed onto zeros, and
/// `moe_tail`, which normalizes each branch, their sum, and adds it through
/// the sublayer's post-norm tail (scaled or not). `norm` is the element of
/// all three tail norms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ParallelBinding {
    pub hidden: u64,
    pub dense: DenseBranchBinding,
    pub routed: crate::operators::routed::GeneralRoutedBinding,
    pub norm: Element,
    pub scaled: bool,
}

/// The dense branch of a [`ParallelBinding`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DenseBranchBinding {
    /// The expansion width (`F`).
    pub features: u64,
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
    pub normalize_selected: bool,
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
    pub head: ReadoutHead,
    pub activation: Element,
}

/// How the plan places the vocabulary projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReadoutHead {
    /// The output projection's resident element and the extent of its
    /// accumulator-scale port.
    Packed { weight: Element, weight_scale: u64 },
    /// Its progressive planes (`ModelLoadPlan::with_progressive_head`).
    Progressive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FeaturesBinding {
    pub norm: Element,
    pub activation: Element,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HeadBinding {
    pub embedding_table: Element,
    pub embedding_norm: Element,
    pub hidden_norm: Element,
    pub combine: Element,
    pub attention: AttentionBinding,
    pub feed_forward: FeedForwardProgramSlot,
    pub output_norm: Element,
    pub projection: HeadProjection,
    pub activation: Element,
}

/// How a draft head projects onto the target's output head.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HeadProjection {
    /// The output projection's resident element.
    Packed(Element),
    /// Its progressive planes (`ModelLoadPlan::with_progressive_head`).
    Progressive,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelLoadPlan {
    pub(super) target: Vec<WeightPlan>,
    pub(super) head: Option<Vec<WeightPlan>>,
    pub(super) vision: Option<Vec<WeightPlan>>,
    /// The target's host-resident gathered tables.
    pub(super) host_tables: Vec<HostTablePlan>,
}

impl ModelLoadPlan {
    pub fn host_tables(&self) -> &[HostTablePlan] {
        &self.host_tables
    }

    /// The host tables' claim in the system-RAM domain.
    pub fn host_table_bytes(&self) -> Result<u64, String> {
        self.host_tables.iter().try_fold(0u64, |total, table| {
            total
                .checked_add(table.bytes)
                .ok_or_else(|| "host table bytes overflow".to_owned())
        })
    }

    /// Peak upload backing while the target component is imported.
    pub fn target_upload_peak_bytes(&self) -> Result<u64, String> {
        let source = self
            .target
            .iter()
            .map(|weight| weight.source_bytes)
            .max()
            .unwrap_or(0);
        source_import_peak_bytes(source)
    }

    /// Peak upload allocation while importing a lazy optional component.
    pub fn head_upload_peak_bytes(&self) -> Result<u64, String> {
        let source = self
            .head
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(|weight| weight.source_bytes)
            .max()
            .unwrap_or(0);
        source_import_peak_bytes(source)
    }

    pub fn vision_upload_peak_bytes(&self) -> Result<u64, String> {
        let source = self
            .vision
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(|weight| weight.source_bytes)
            .max()
            .unwrap_or(0);
        source_import_peak_bytes(source)
    }
}

pub(super) fn source_import_peak_bytes(source_bytes: u64) -> Result<u64, String> {
    if source_bytes == 0 {
        return Ok(0);
    }
    // The importer learns the source file offset at load. Bound a host-page
    // aligned mapping with both a leading and a trailing partial page.
    #[cfg(unix)]
    {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let page =
            u64::try_from(page).map_err(|_| "host page size is unavailable for import planning")?;
        source_bytes
            .checked_add(page.checked_mul(2).ok_or("import page bound overflow")?)
            .ok_or_else(|| "import window bound overflow".into())
    }
    #[cfg(not(unix))]
    {
        Ok(source_bytes)
    }
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
            let storage = (
                weight.upload,
                weight.shape.clone(),
                weight.source_bytes,
                weight.resident_bytes,
                weight.scale.clone(),
            );
            match seen.entry(weight.storage_identity()) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(storage);
                    bytes[index] = bytes[index]
                        .checked_add(weight.storage_bytes()?)
                        .ok_or("resident weight byte count overflow")?;
                }
                std::collections::hash_map::Entry::Occupied(entry) if entry.get() != &storage => {
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
        history: KvCodec,
    ) -> Result<ProgramPlan, PlanError> {
        super::programs::derive_program_plan(
            definition,
            &self.target,
            self.head.as_deref(),
            self.vision.as_deref(),
            &self.host_tables,
            history,
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
        Self::derive_components(
            manifest.identity,
            &manifest.target.tensors,
            manifest
                .projector
                .as_ref()
                .map(|projector| (projector.identity, projector.tensors.as_slice())),
            manifest
                .draft
                .as_ref()
                .map(|draft| (draft.identity, draft.tensors.as_slice())),
            definition,
            selection,
            layout,
        )
    }

    /// Plan resident representations from a pre-download header bundle.
    /// The resulting plan describes bytes and formats but cannot import
    /// weights until a payload-backed `Package` is opened separately.
    pub fn derive_headers(
        headers: &PackageHeaders,
        definition: &ModelDefinition,
        selection: ComponentSelection,
        layout: Layout,
    ) -> Result<Self, String> {
        let identity = headers.identity();
        let projector = headers
            .projector()
            .map(|directory| {
                identity
                    .projector
                    .map(|component| (component, directory.tensors.as_slice()))
                    .ok_or_else(|| "projector header has no component identity".to_owned())
            })
            .transpose()?;
        let manifest = headers.manifest();
        Self::derive_components(
            identity,
            &headers.target().tensors,
            projector,
            manifest
                .draft
                .as_ref()
                .map(|draft| (draft.identity, draft.tensors.as_slice())),
            definition,
            selection,
            layout,
        )
    }

    fn derive_components(
        identity: PackageIdentity,
        target_tensors: &[TensorDescriptor],
        projector: Option<(ArtifactIdentity, &[TensorDescriptor])>,
        draft: Option<(ArtifactIdentity, &[TensorDescriptor])>,
        definition: &ModelDefinition,
        selection: ComponentSelection,
        layout: Layout,
    ) -> Result<Self, String> {
        definition.validate().map_err(|error| error.to_string())?;
        if identity != definition.artifact_identity {
            return Err("load plan package identity mismatch".into());
        }
        let target_component = ArtifactComponent {
            kind: ArtifactComponentKind::Target,
            identity: identity.target,
        };
        let decoder = &definition.decoder;
        let target_form = ResidentForm {
            activation: activation_dtype(decoder.activation_dtype),
            layout,
        };
        let target_role = |kind| WeightRole {
            scope: WeightScope::Target,
            kind,
        };
        let mut target = Vec::new();
        // A per-layer entry projects the embedding on the device; its table
        // is a host table.
        let per_layer_weights = decoder.entry.per_layer.iter().flat_map(|entry| {
            [
                (
                    target_role(WeightKind::PerLayerModelProjection),
                    &entry.projection,
                ),
                (
                    target_role(WeightKind::PerLayerProjectionNorm),
                    &entry.projection_norm.weight,
                ),
            ]
        });
        let host_tables = decoder
            .entry
            .per_layer
            .iter()
            .map(|entry| {
                host_table(
                    target_tensors,
                    target_component,
                    target_role(WeightKind::PerLayerTable),
                    &entry.table,
                    target_form,
                )
            })
            .collect::<Result<Vec<_>, String>>()?;
        for (role, descriptor) in std::iter::once((
            target_role(WeightKind::Embedding),
            &decoder.entry.embedding,
        ))
        .chain(per_layer_weights)
        .chain(operators::decoder_weights(decoder))
        .chain([
            (target_role(WeightKind::OutputNorm), decoder.exit.norm.weight()),
            (target_role(WeightKind::Output), &decoder.exit.output),
        ]) {
            push_weight(
                &mut target,
                target_tensors,
                target_component,
                role.scope,
                role.kind,
                descriptor,
                target_form,
            )?;
        }

        // The executed definition carries at most one drafter.
        if selection.head && definition.head.is_some() == definition.draft.is_some() {
            return Err(
                "head execution was selected without exactly one drafter definition".into(),
            );
        }
        if projector.is_some() != definition.vision.is_some() {
            return Err("projector component and vision definition disagree".into());
        }
        if draft.is_some() != definition.draft.is_some() {
            return Err("draft component and draft definition disagree".into());
        }
        if selection.vision && definition.vision.is_none() {
            return Err("vision execution was selected without a vision definition".into());
        }
        let push_all = |plans: &mut Vec<WeightPlan>,
                        tensors: &[TensorDescriptor],
                        component: ArtifactComponent,
                        weights: Vec<(WeightRole, &WeightDescriptor)>| {
            for (role, descriptor) in weights {
                push_weight(
                    plans,
                    tensors,
                    component,
                    role.scope,
                    role.kind,
                    descriptor,
                    target_form,
                )?;
            }
            Ok::<_, String>(())
        };
        let head = match (selection.head, &definition.head, &definition.draft, draft) {
            (false, ..) => None,
            (true, Some(head), None, _) => {
                let mut plans = Vec::new();
                push_all(
                    &mut plans,
                    target_tensors,
                    target_component,
                    operators::head_weights(head).map_err(|error| error.to_string())?,
                )?;
                Some(plans)
            }
            (true, None, Some(separate), Some((identity, tensors))) => {
                let component = ArtifactComponent {
                    kind: ArtifactComponentKind::Draft,
                    identity,
                };
                // The fusion is read by every target step, so it is resident
                // with the target; the drafter's own weights load with it.
                push_all(
                    &mut target,
                    tensors,
                    component,
                    operators::draft::fusion_weights(separate).to_vec(),
                )?;
                let mut plans = Vec::new();
                push_all(
                    &mut plans,
                    tensors,
                    component,
                    operators::draft::draft_weights(separate).map_err(|error| error.to_string())?,
                )?;
                Some(plans)
            }
            _ => return Err("the selected drafter has no component".into()),
        };

        let vision = selection
            .vision
            .then_some(definition.vision.as_ref())
            .flatten()
            .map(|vision| {
                let (projector_identity, projector_tensors) =
                    projector.ok_or("vision definition requires a projector component")?;
                let component = ArtifactComponent {
                    kind: ArtifactComponentKind::Projector,
                    identity: projector_identity,
                };
                plan_vision(projector_tensors, component, vision, layout)
            })
            .transpose()?;
        validate_unique_roles(
            target
                .iter()
                .chain(head.as_deref().unwrap_or_default())
                .chain(vision.as_deref().unwrap_or_default()),
        )?;

        Ok(Self {
            target,
            head,
            vision,
            host_tables,
        })
    }

    /// Place the output projection in progressive planes
    /// (`ImportTransform::Progressive`, `WeightKind::OutputPlane`), replacing
    /// it, when the head admits them: a Q8_0 matrix of whole groups, imported
    /// as stored without transforms or a second-level scale, a decoder
    /// without a logit softcap, and no selected drafter (a drafter projects
    /// through the stored head). Otherwise the plan is unchanged.
    pub fn with_progressive_head(mut self, definition: &ModelDefinition) -> Result<Self, String> {
        let role = WeightRole {
            scope: WeightScope::Target,
            kind: WeightKind::Output,
        };
        let index = self
            .target
            .iter()
            .position(|weight| weight.role == role)
            .ok_or("the plan has no output projection")?;
        let output = &self.target[index];
        // A draft head projects onto the planes too; a separate draft's
        // readouts read the packed output projection.
        let drafts = self
            .target
            .iter()
            .any(|weight| weight.role.scope == WeightScope::Draft);
        let admitted = !drafts
            && output.scale.is_none()
            && output.descriptor.transforms.is_empty()
            && output.upload == output.source
            && Some(output.source) == source_element(Encoding::Q8_0)
            && output.shape.len() == 2
            && output.shape[1].is_multiple_of(32)
            && definition.decoder.exit.softcap.is_none_or(|cap| cap == 0.0);
        if !admitted {
            return Ok(self);
        }
        let planes = ProgressivePlane::ALL
            .into_iter()
            .map(|plane| {
                let descriptor = WeightDescriptor {
                    transforms: vec![ImportTransform::Progressive(plane)],
                    ..output.descriptor.clone()
                };
                let shape = descriptor
                    .transformed_shape(&output.shape)
                    .map_err(|error| error.to_string())?;
                let element = crate::progressive::element(plane);
                Ok(WeightPlan {
                    role: WeightRole {
                        scope: WeightScope::Target,
                        kind: WeightKind::OutputPlane(plane),
                    },
                    component: output.component,
                    source: output.source,
                    upload: element,
                    resident: element,
                    source_bytes: representation_bytes(element, &shape)?,
                    resident_bytes: representation_bytes(element, &shape)?,
                    descriptor: WeightDescriptor {
                        shape: shape.clone(),
                        ..descriptor
                    },
                    shape,
                    scale: None,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        self.target.splice(index..=index, planes);
        Ok(self)
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

/// The dense element dtype of a decoder's activations.
pub(crate) fn activation_dtype(dtype: ActivationDType) -> DType {
    match dtype {
        ActivationDType::F16 => DType::F16,
        ActivationDType::BF16 => DType::BF16,
    }
}

/// The per-layer entry (Gemma PLE, `EntryForm.per_layer`): the batch rows'
/// host-table rows (uploaded as `table_source`, converted to `table`) and
/// the embedding's projection to `layers · width` channels, combined by
/// `per_layer_inputs` into every block's per-layer input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PerLayerEntryBinding {
    pub hidden: u64,
    pub layers: u64,
    pub width: u64,
    pub table_source: Element,
    pub table: Element,
    pub projection: Element,
    pub norm: Element,
    pub activation: Element,
}

/// A block's per-layer input sublayer: `per_layer_gate` over its layer's
/// slice, `project_rows` back to the hidden width, and its post-norm tail.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PerLayerBinding {
    pub hidden: u64,
    pub layers: u64,
    pub width: u64,
    pub gate: Element,
    pub projection: Element,
    pub activation: Element,
    /// Always a post-norm tail.
    pub tail: SublayerTail,
}

/// A host-resident gathered table (model-family plan §3.7): read by row
/// gather from its artifact and never a device weight. Each step uploads the
/// batch rows' `source` rows, which the graph converts to `resident` rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostTablePlan {
    pub role: WeightRole,
    pub component: ArtifactComponent,
    pub descriptor: WeightDescriptor,
    /// `[rows, columns]`.
    pub shape: Vec<u64>,
    pub source: Element,
    pub resident: Element,
    /// The table's bytes: its claim in the system-RAM domain.
    pub bytes: u64,
}

impl HostTablePlan {
    pub fn columns(&self) -> u64 {
        self.shape[1]
    }
}

/// Plan the host table `descriptor` names: a stored `[rows, columns]`
/// matrix without import transforms, in a source encoding whose rows the
/// backend layout can convert on the device.
fn host_table(
    inventory: &[TensorDescriptor],
    component: ArtifactComponent,
    role: WeightRole,
    descriptor: &WeightDescriptor,
    form: ResidentForm,
) -> Result<HostTablePlan, String> {
    let stored = inventory
        .iter()
        .find(|tensor| tensor.name == descriptor.name)
        .ok_or_else(|| format!("host table {:?} is absent from its component", descriptor.name))?;
    if !descriptor.transforms.is_empty() || stored.shape != descriptor.shape {
        return Err(format!(
            "host table {:?} must be the stored matrix itself",
            descriptor.name
        ));
    }
    let [_, _] = stored.shape[..] else {
        return Err(format!("host table {:?} is not a matrix", descriptor.name));
    };
    let source = source_element(stored.encoding)
        .ok_or_else(|| format!("unsupported source encoding {:?}", stored.encoding))?;
    let resident = resident_element(stored.encoding, form.activation, form.layout)
        .ok_or_else(|| {
            format!(
                "{:?} has no resident form in the `{}` layout",
                stored.encoding,
                form.layout.as_str()
            )
        })?;
    let bytes = representation_bytes(source, &stored.shape)?;
    if bytes != stored.nbytes {
        return Err(format!(
            "host table {:?} stores {} bytes, but its {:?} shape {:?} is {bytes}",
            descriptor.name, stored.nbytes, stored.encoding, stored.shape
        ));
    }
    Ok(HostTablePlan {
        role,
        component,
        descriptor: descriptor.clone(),
        shape: stored.shape.clone(),
        source,
        resident,
        bytes,
    })
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
    // The logical shape the importer's exact transforms make of the stored
    // tensor (the stored shape when there are none).
    let logical = crate::import_transforms::admit(descriptor, &stored.shape, stored.encoding)?;
    let role = WeightRole { scope, kind };
    // Projector weights keep their stored dense element: the vision kernels
    // read f32, f16 and bf16 matrices and vectors directly, and converting
    // f16 matrices or f32 biases to the activation dtype only loses bits.
    let stored_dense = source_element(stored.encoding).and_then(Element::dtype);
    let dense_resident = match (role.kind, stored_dense) {
        (WeightKind::Vision(_), Some(dtype)) => dtype,
        _ => resident_dtype(role, form.activation),
    };
    if operators::is_fixed_dense_role(kind)
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
    // Packed projector weights become dense in the activation dtype: the GPU
    // vision kernels read dense weights only. The host dequantizes them
    // exactly to F32 and the device rounds once to the activation dtype.
    let (upload, resident) = match (role.kind, stored_dense) {
        (WeightKind::Vision(_), None) => {
            (Element::f32(), Element::dense(form.activation))
        }
        _ => (
            source,
            resident_element(stored.encoding, dense_resident, form.layout).ok_or_else(|| {
                format!(
                    "{:?} has no resident form in the `{}` layout",
                    stored.encoding,
                    form.layout.as_str()
                )
            })?,
        ),
    };
    let source_bytes = representation_bytes(source, &stored.shape)?;
    if source_bytes != stored.nbytes {
        return Err(format!(
            "weight {:?} ({role:?}) stores {} bytes, but its {:?} shape {:?} is {source_bytes}",
            descriptor.name, stored.nbytes, stored.encoding, stored.shape
        ));
    }
    let resident_bytes = representation_bytes(resident, &logical)?;
    let scale = weight_scale(inventory, descriptor, &logical)?;
    if scale.is_some() && upload != source {
        return Err(format!(
            "weight {:?} ({role:?}) is dequantized for dense kernels, which have no \
             accumulator-scale port for its second-level scale",
            descriptor.name
        ));
    }
    out.push(WeightPlan {
        role,
        component,
        source,
        upload,
        resident,
        shape: logical.clone(),
        descriptor: descriptor.clone(),
        // The bytes the importer uploads: the transformed source, in the
        // upload representation.
        source_bytes: representation_bytes(upload, &logical)?,
        resident_bytes,
        scale,
    });
    Ok(())
}

/// The resident second-level scale of a weight whose transforms name one
/// (`import_transforms::admit` allows at most one): a stored F32 tensor of
/// one value, or one per matrix of a stacked `logical` weight.
fn weight_scale(
    inventory: &[TensorDescriptor],
    descriptor: &WeightDescriptor,
    logical: &[u64],
) -> Result<Option<WeightScalePlan>, String> {
    let Some(tensor) = crate::import_transforms::scale_tensor(descriptor) else {
        return Ok(None);
    };
    let stored = inventory
        .iter()
        .find(|stored| stored.name == tensor)
        .ok_or_else(|| {
            format!(
                "second-level scale {tensor:?} of {:?} is absent from its component",
                descriptor.name
            )
        })?;
    let matrices = match logical {
        [stack, _, _] => *stack,
        [_, _] => 1,
        _ => {
            return Err(format!(
                "second-level scale {tensor:?} scales {:?}, which is not a matrix or a stack of \
                 matrices",
                descriptor.name
            ))
        }
    };
    let [extent] = stored.shape[..] else {
        return Err(format!(
            "second-level scale {tensor:?} of {:?} has shape {:?}, not [1] or [{matrices}]",
            descriptor.name, stored.shape
        ));
    };
    if stored.encoding != Encoding::F32 || !(extent == 1 || extent == matrices) {
        return Err(format!(
            "second-level scale {tensor:?} of {:?} is {:?} {:?}, not F32 [1] or [{matrices}]",
            descriptor.name, stored.encoding, stored.shape
        ));
    }
    if representation_bytes(Element::f32(), &stored.shape)? != stored.nbytes {
        return Err(format!(
            "second-level scale {tensor:?} stores {} bytes for its shape {:?}",
            stored.nbytes, stored.shape
        ));
    }
    Ok(Some(WeightScalePlan {
        tensor: tensor.to_owned(),
        stored: extent,
        extent: matrices,
        resident_bytes: representation_bytes(Element::f32(), &[matrices])?,
    }))
}

/// The resident dense element of a role (`operators::resident_dtype`).
pub(super) fn resident_dtype(role: WeightRole, activation: DType) -> DType {
    operators::resident_dtype(role.kind, activation)
}

fn plan_vision(
    inventory: &[TensorDescriptor],
    component: ArtifactComponent,
    vision: &VisionDescription,
    layout: Layout,
) -> Result<Vec<WeightPlan>, String> {
    let form = ResidentForm {
        activation: activation_dtype(vision.activation_dtype),
        layout,
    };
    let mut out = Vec::new();
    for (role, descriptor) in vision.weights() {
        push_weight(
            &mut out, inventory, component, role.scope, role.kind, descriptor, form,
        )?;
    }
    Ok(out)
}

/// The planned element of a role an entry binds without an accumulator-scale
/// port: a weight with a second-level scale is refused.
pub(super) fn planned_element(
    plans: &[WeightPlan],
    scope: WeightScope,
    kind: WeightKind,
) -> Result<Element, PlanError> {
    let weight = planned_scalable(plans, scope, kind)?;
    if weight.scale != 0 {
        return Err(PlanError::UnportedScale(WeightRole { scope, kind }));
    }
    Ok(weight.element)
}

/// The planned form of a role an entry binds with an accumulator-scale port.
pub(super) fn planned_scalable(
    plans: &[WeightPlan],
    scope: WeightScope,
    kind: WeightKind,
) -> Result<ScalableWeight, PlanError> {
    plans
        .iter()
        .find(|plan| plan.role == WeightRole { scope, kind })
        .map(|plan| ScalableWeight {
            element: plan.resident,
            scale: plan.scale_extent(),
        })
        .ok_or_else(|| {
            PlanError::InvalidDefinition(format!("load plan is missing {scope:?}/{kind:?}"))
        })
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
/// Metal `rows32`, Vulkan `rows16`, CUDA `mma16`; native CPU kernels read `rows8`, while every planned
/// (compiled) kernel reads the `packet` layout.
pub fn resident_layout(path: ExecutionPath, backend: BackendName) -> Layout {
    match (path, backend) {
        (ExecutionPath::Native, BackendName::Metal) => Layout::Rows32,
        (ExecutionPath::Native, BackendName::Vulkan) => Layout::Rows16,
        (ExecutionPath::Native, BackendName::Cuda) => Layout::Mma16,
        (ExecutionPath::Native, BackendName::Cpu) => Layout::Rows8,
        (ExecutionPath::Planned, _) => Layout::Packet,
    }
}

/// The external storage of an artifact encoding.
pub fn source_element(encoding: Encoding) -> Option<Element> {
    match encoding {
        Encoding::F32 => Some(Element::dense(DType::F32)),
        Encoding::F16 => Some(Element::dense(DType::F16)),
        Encoding::BF16 => Some(Element::dense(DType::BF16)),
        Encoding::Q8_0 => Element::named("gguf_q8_0"),
        Encoding::Q3K => Element::named("gguf_q3_k"),
        Encoding::Q4K => Element::named("gguf_q4_k"),
        Encoding::Q5K => Element::named("gguf_q5_k"),
        Encoding::Q6K => Element::named("gguf_q6_k"),
        Encoding::Iq3S => Element::named("gguf_iq3_s"),
        Encoding::Iq4Nl => Element::named("gguf_iq4_nl"),
        Encoding::Iq4Xs => Element::named("gguf_iq4_xs"),
        Encoding::Q4_0 => Element::named("gguf_q4_0"),
        Encoding::Q5_0 => Element::named("gguf_q5_0"),
        Encoding::Q5_1 => Element::named("gguf_q5_1"),
        Encoding::Mxfp4 => Element::named("gguf_mxfp4"),
        Encoding::Nvfp4 => Element::named("gguf_nvfp4"),
        _ => None,
    }
}

/// The one map from an artifact encoding to resident storage: dense
/// encodings become `dense`; packed encodings become their representation
/// (from the source format) in `layout` (from the device backend). A
/// format without its own representation imports exactly into one that
/// holds every value it encodes: Q3_K and IQ3_S into q6k (an f16
/// super-scale times an int8 sixteen-value sub-block scale times a code in
/// [-32, 31]), IQ4_NL into iq4g32 (same table). NVFP4's per-tensor F32
/// `.scale` is a separate tensor; its representation holds the block values.
pub fn resident_element(encoding: Encoding, dense: DType, layout: Layout) -> Option<Element> {
    let representation = match encoding {
        Encoding::F32 | Encoding::F16 | Encoding::BF16 => return Some(Element::dense(dense)),
        Encoding::Q8_0 => "q8g32s",
        Encoding::Q3K | Encoding::Iq3S | Encoding::Q6K => "q6k",
        Encoding::Q4K => "q4k",
        Encoding::Q5K => "q5k",
        Encoding::Iq4Nl | Encoding::Iq4Xs => "iq4g32",
        Encoding::Q4_0 => "q4g32s",
        Encoding::Q5_0 => "q5g32s",
        Encoding::Q5_1 => "q5g32",
        Encoding::Mxfp4 => "mxfp4g32",
        Encoding::Nvfp4 => "nvfp4g16",
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
            (ExecutionPath::Native, BackendName::Metal, Layout::Rows32),
            (ExecutionPath::Native, BackendName::Vulkan, Layout::Rows16),
            (ExecutionPath::Native, BackendName::Cuda, Layout::Mma16),
            (ExecutionPath::Native, BackendName::Cpu, Layout::Rows8),
            (ExecutionPath::Planned, BackendName::Metal, Layout::Packet),
        ] {
            assert_eq!(resident_layout(path, backend), layout);
        }
        for (encoding, representation) in [
            (Encoding::Q8_0, "q8g32s"),
            (Encoding::Q3K, "q6k"),
            (Encoding::Q4K, "q4k"),
            (Encoding::Q5K, "q5k"),
            (Encoding::Q6K, "q6k"),
            (Encoding::Iq3S, "q6k"),
            (Encoding::Iq4Nl, "iq4g32"),
            (Encoding::Iq4Xs, "iq4g32"),
            (Encoding::Q4_0, "q4g32s"),
            (Encoding::Q5_0, "q5g32s"),
            (Encoding::Q5_1, "q5g32"),
            (Encoding::Mxfp4, "mxfp4g32"),
            (Encoding::Nvfp4, "nvfp4g16"),
        ] {
            for layout in Layout::ALL {
                let element = resident_element(encoding, DType::BF16, layout).unwrap();
                assert_eq!(
                    (element.representation(), element.layout()),
                    (representation, layout)
                );
            }
        }
        assert_eq!(
            resident_element(Encoding::F16, DType::BF16, Layout::Rows16),
            Some(Element::bf16())
        );
        for encoding in [Encoding::Q1_0, Encoding::I32] {
            assert_eq!(
                resident_element(encoding, DType::BF16, Layout::Rows16),
                None
            );
        }
    }
}
