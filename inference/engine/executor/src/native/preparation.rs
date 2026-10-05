use super::specialization::{Specializer, Tuning};
use super::tuning::{
    attention::{
        AttentionDecodeK8V4Tuning, AttentionDecodeTuning, AttentionMix, AttentionOutputTuning,
        AttentionPrefillK8V4Tuning, AttentionPrefillTuning, AttentionProjectTuning,
    },
    cases::{DenseExpandTuning, DenseOutputTuning, DenseUpTuning},
    general_routed::{
        routing_shape, RoutedDownTuning, RoutedExpandDecodeTuning, RoutedExpertTilesTuning,
        RoutedGateUpTuning, RoutedGatedTilesTuning, RoutedScatterTuning, RoutedSelectTuning,
        RoutedUpTilesTuning, RoutedUpTuning,
    },
    post_norm::ProjectRowsTuning,
    readout::{
        DraftRowsTuning, ExactRowsTuning, HeadLogitsTuning, HeadRowsTuning, PlanesRowsTuning,
        ProgressiveTuning, RefineRowsTuning, SampleRowsTuning, SelectedRowsTuning, ShapeRowsTuning,
        TopRowsTuning,
    },
    recurrent::{
        RecurrentChunkTuning, RecurrentOutputTuning, RecurrentProjectConvolvedTuning,
        RecurrentProjectTuning, RecurrentShape, RecurrentState, RecurrentStepConvolvedTuning,
        RecurrentStepTuning,
    },
    routed::{
        RoutedChoicesTuning, RoutedCombineTuning, RoutedExpandTuning, RoutedExpertsTuning,
        RoutedGroupTuning, RoutedOutputTuning, RoutedRouteSharedTuning, RoutedRouteTuning,
        RoutedShape,
    },
    short_conv::{ShortConvOutputTuning, ShortConvProjectTuning},
    state_space::{
        StateSpaceChunkTuning, StateSpaceOutputTuning, StateSpaceProjectTuning, StateSpaceState,
        StateSpaceStepTuning,
    },
    ModelInputs, TunedEntry, Tuner, TuningLimits, TuningWeights,
};
use super::*;
use crate::operators::gated_delta::graph::StepForm;
use crate::operators::routed::fused_graph::DecodeForm;
use crate::{
    DenseBinding, GeneralRoutedBinding, HeadProjection, ModelLoadPlan, ReadoutHead, ShortConvBinding,
    StateSpaceBinding, SublayerTail,
};
use magnitude_family_contracts::{ModelDefinition, SublayerIndex, WeightKind, WeightScope};
use magnitude_kernels::{
    conditioning_overlay, draft_confidence, draft_convolve_input, draft_convolve_residual,
    draft_gated_rows, draft_path_step, draft_top_k, feature_rows, import_dense, moe_tail,
    per_layer_inputs, post_norm_residual, repack_weight, tap_rows, widen_rows,
};
use magnitude_state::KvCodec;
use seismic::KernelRequest;
use std::collections::HashSet;

/// The phase-one catalog. Every handle is prepared before qualification and
/// retained for warm calls; this type has no API capable of preparing again.
#[derive(Debug)]
pub(super) struct NativePreparationCache {
    pub(super) owner: Tensor,
    pub(super) import: ImportKernels,
    pub(super) glue: GlueKernels,
    pub(super) target: TargetKernels,
    pub(super) head: Option<HeadKernels>,
    pub(super) draft: Option<DraftKernels>,
    pub(super) vision: Option<VisionKernels>,
    pub(super) tuned: Vec<TunedEntry>,
}

/// Everything program preparation reads.
pub(super) struct PreparationInputs<'a> {
    pub device: &'a Device,
    pub plan: &'a ProgramPlan,
    pub load: &'a ModelLoadPlan,
    pub limits: TuningLimits,
    pub tuning: TuningContext<'a>,
}

/// Prepare one entry that has no tuning case. Evaluates to `Option`: `None`
/// during a count or a listing.
macro_rules! fixed {
    ($spec:expr, $module:ident, $bindings:expr, $elements:expr) => {
        fixed!($spec, $module, $bindings, $elements, statics & [])
    };
    ($spec:expr, $module:ident, $bindings:expr, $elements:expr, statics $statics:expr) => {
        $spec.fixed(&$bindings, $statics, $module::native_entry_with($elements))?
    };
    ($spec:expr, $module:ident, $bindings:expr) => {
        $spec.fixed(&$bindings, &[], $module::native_entry())?
    };
}

/// The layers each binding is prepared for, in model order. Tuning cases
/// rotate through these layers' resident weights.
struct BindingLayers<K> {
    layers: HashMap<K, Vec<WeightScope>>,
}

impl<K: Copy + Eq + std::hash::Hash> BindingLayers<K> {
    fn of(bindings: impl IntoIterator<Item = (K, WeightScope)>) -> Self {
        let mut layers: HashMap<K, Vec<WeightScope>> = HashMap::new();
        for (binding, scope) in bindings {
            layers.entry(binding).or_default().push(scope);
        }
        Self { layers }
    }

    fn scopes(&self, binding: K) -> Vec<WeightScope> {
        self.layers[&binding].clone()
    }
}

/// The scope of sublayer `sublayer` (0 the mixer, 1 the feed-forward) of
/// target block `index`.
fn sublayer_scope(index: usize, sublayer: u32) -> WeightScope {
    WeightScope::TargetSublayer(SublayerIndex {
        block: u32::try_from(index).expect("block count fits u32"),
        sublayer,
    })
}

fn head_scope(index: usize) -> WeightScope {
    WeightScope::HeadBlock(u32::try_from(index).expect("head block count fits u32"))
}

fn head_sublayer_scope(index: usize, sublayer: u32) -> WeightScope {
    WeightScope::HeadSublayer(SublayerIndex {
        block: u32::try_from(index).expect("head block count fits u32"),
        sublayer,
    })
}

impl NativePreparationCache {
    pub(super) fn prepare_programs(inputs: PreparationInputs<'_>) -> Result<Self, CatalogFailure> {
        let PreparationInputs {
            device,
            plan,
            load,
            limits,
            tuning,
        } = inputs;
        let mut spec = Specializer::new(device);
        let import = imports(&mut spec, plan)?;
        let owner = Tensor::zeros(device, Element::u32(), &[1]).map_err(|error| {
            CatalogFailure::Preparation {
                entry: "catalog_owner",
                bindings: "A=u32".into(),
                outcome: error.to_string(),
            }
        })?;
        let epsilon = epsilon(tuning.definition)?;
        // Tuning walks the program three times: a count finds the units that
        // will search and their launches, a census measures their defaults
        // at their points within the tuning time left (keeping the inputs it
        // builds), and
        // the last walk searches each within its share of step time and
        // prepares it.
        let mut count = Preparation::new(
            plan,
            tuning.definition,
            limits,
            epsilon,
            Specializer::count(device),
            Tuning::Tuner(Tuner::count(
                device,
                tuning,
                limits,
                TuningWeights::new(device, load, tuning.weights, &import),
            )),
        );
        count.walk(plan)?;
        let mut census = Preparation::new(
            plan,
            tuning.definition,
            limits,
            epsilon,
            Specializer::count(device),
            Tuning::Tuner(count.tuning.into_tuner().census()),
        );
        census.walk(plan)?;
        let mut preparation = Preparation::new(
            plan,
            tuning.definition,
            limits,
            epsilon,
            spec,
            Tuning::Tuner(census.tuning.into_tuner().search()),
        );
        let glue = preparation.walk(plan)?;
        let Preparation {
            tuning,
            target,
            head,
            draft,
            vision,
            ..
        } = preparation;
        let tuned = tuning.into_tuner().tuned();
        Ok(Self {
            owner,
            import,
            glue,
            target,
            head,
            draft,
            vision,
            tuned,
        })
    }
}

/// The requests of the native kernels preparing `plan` on `backend` prepares,
/// named without a device: the weight-import entries and the entries the
/// program walk prepares, each once, in walk order.
pub(crate) fn kernel_requests(
    backend: BackendName,
    plan: &ProgramPlan,
    definition: &ModelDefinition,
    load: &ModelLoadPlan,
    limits: TuningLimits,
) -> Result<Vec<KernelRequest>, CatalogFailure> {
    let mut spec = Specializer::list(backend);
    imports(&mut spec, plan)?;
    let mut listing = Preparation::new(
        plan,
        definition,
        limits,
        epsilon(definition)?,
        spec,
        Tuning::Listing(ModelInputs {
            definition,
            limits,
            load,
        }),
    );
    listing.walk(plan)?;
    let mut seen = HashSet::new();
    Ok(listing
        .spec
        .into_requests()
        .into_iter()
        .filter(|request| seen.insert(request.clone()))
        .collect())
}

/// Every admitted definition normalizes with one epsilon, which the
/// readout's final norm states.
fn epsilon(definition: &ModelDefinition) -> Result<f32, CatalogFailure> {
    crate::programs::graph::readout::readout_epsilon(&definition.decoder).map_err(|outcome| {
        CatalogFailure::Preparation {
            entry: "normalization",
            bindings: "decoder epsilon".into(),
            outcome,
        }
    })
}

/// Prepare the weight-import entries of `plan`'s import slots.
fn imports(spec: &mut Specializer<'_>, plan: &ProgramPlan) -> Result<ImportKernels, CatalogFailure> {
    let mut import = ImportKernels {
        import_dense: HashMap::new(),
        repack_weight: HashMap::new(),
    };
    for slot in plan.imports() {
        match *slot {
            ImportProgramSlot::Dense { source, resident } => {
                if import.import_dense.contains_key(&(source, resident)) {
                    continue;
                }
                let bindings = dense_binding_name(source, resident);
                if let Some(kernel) = fixed!(
                    spec,
                    import_dense,
                    bindings,
                    import_dense::Elements {
                        E: Element::dense(source),
                        U: Element::dense(resident),
                    }
                ) {
                    import.import_dense.insert((source, resident), kernel);
                }
            }
            ImportProgramSlot::Repack { source, resident } => {
                if import.repack_weight.contains_key(&(source, resident)) {
                    continue;
                }
                let bindings = element_binding_name(source, resident);
                if let Some(kernel) = fixed!(
                    spec,
                    repack_weight,
                    bindings,
                    repack_weight::Elements {
                        E: source,
                        U: resident,
                    }
                ) {
                    import.repack_weight.insert((source, resident), kernel);
                }
            }
        }
    }
    Ok(import)
}

struct Preparation<'a> {
    limits: TuningLimits,
    spec: Specializer<'a>,
    tuning: Tuning<'a>,
    epsilon: f32,
    /// Model width: the static dimension of the feature readout.
    hidden: u64,
    /// The target's vocabulary: the static dimension of token selection.
    vocabulary: u64,
    target: TargetKernels,
    head: Option<HeadKernels>,
    draft: Option<DraftKernels>,
    vision: Option<VisionKernels>,
}

impl<'a> Preparation<'a> {
    fn new(
        plan: &ProgramPlan,
        definition: &ModelDefinition,
        limits: TuningLimits,
        epsilon: f32,
        spec: Specializer<'a>,
        tuning: Tuning<'a>,
    ) -> Self {
        Self {
            limits,
            spec,
            tuning,
            epsilon,
            hidden: definition.decoder.hidden,
            vocabulary: definition.decoder.vocabulary,
            target: TargetKernels::default(),
            head: plan.head().map(|_| HeadKernels::default()),
            draft: plan.draft().map(|_| DraftKernels::default()),
            vision: plan.vision().map(|_| VisionKernels::default()),
        }
    }

    /// Prepare every program entry, in a fixed order.
    fn walk(&mut self, plan: &ProgramPlan) -> Result<GlueKernels, CatalogFailure> {
        let glue = self.glue(plan)?;
        self.target(plan)?;
        self.head(plan)?;
        self.draft(plan)?;
        self.vision(plan)?;
        Ok(glue)
    }

    /// Token selection, the conditioning overlay and the state row copies.
    fn glue(&mut self, plan: &ProgramPlan) -> Result<GlueKernels, CatalogFailure> {
        let copies = plan.state().copies();
        let copy = |spec: &mut Specializer<'_>, element: Element, bindings: &'static str| {
            if !copies.contains(&element) {
                return Ok(None);
            }
            Ok::<_, CatalogFailure>(fixed!(
                spec,
                copy_rows,
                bindings,
                copy_rows::Elements { A: element }
            ))
        };
        Ok(GlueKernels {
            shape_rows: self.spec.tuned(
                &mut self.tuning,
                &ShapeRowsTuning {
                    vocabulary: self.vocabulary,
                },
            )?,
            sample_rows: self.spec.tuned(
                &mut self.tuning,
                &SampleRowsTuning {
                    vocabulary: self.vocabulary,
                },
            )?,
            conditioning_overlay: fixed!(self.spec, conditioning_overlay, "fixed"),
            copy_rows_f32: copy(&mut self.spec, Element::f32(), "A=f32")?,
            copy_rows_f16: copy(&mut self.spec, Element::f16(), "A=f16")?,
            copy_rows_bf16: copy(&mut self.spec, Element::bf16(), "A=bf16")?,
            copy_rows_u32: copy(&mut self.spec, Element::u32(), "A=u32")?,
        })
    }

    fn target(&mut self, plan: &ProgramPlan) -> Result<(), CatalogFailure> {
        let spec = &mut self.spec;
        let target = plan.target();
        let b = target.embedding();
        if let Some(kernel) = spec.fixed(
            &format!("{b:?}"),
            &[("D", self.hidden)],
            embedding_rows::native_entry_with(embedding_rows::Elements {
                EW: b.table,
                A: b.activation,
            }),
        )? {
            self.target.embedding.insert(b, kernel);
        }
        let attention_layers = BindingLayers::of(target.blocks().iter().enumerate().filter_map(
            |(index, block)| match block.mixer() {
                MixerProgramSlot::Attention(binding) => Some((binding, sublayer_scope(index, 0))),
                _ => None,
            },
        ));
        let recurrent_layers = BindingLayers::of(target.blocks().iter().enumerate().filter_map(
            |(index, block)| match block.mixer() {
                MixerProgramSlot::Recurrent(binding) => Some((binding, sublayer_scope(index, 0))),
                _ => None,
            },
        ));
        let state_space_layers = BindingLayers::of(target.blocks().iter().enumerate().filter_map(
            |(index, block)| match block.mixer() {
                MixerProgramSlot::StateSpace(binding) => Some((binding, sublayer_scope(index, 0))),
                _ => None,
            },
        ));
        let short_conv_layers = BindingLayers::of(target.blocks().iter().enumerate().filter_map(
            |(index, block)| match block.mixer() {
                MixerProgramSlot::ShortConv(binding) => Some((binding, sublayer_scope(index, 0))),
                _ => None,
            },
        ));
        let dense_layers = BindingLayers::of(target.blocks().iter().enumerate().filter_map(
            |(index, block)| match block.feed_forward()? {
                FeedForwardProgramSlot::Dense(binding) => Some((binding, sublayer_scope(index, 1))),
                FeedForwardProgramSlot::Routed(_)
                | FeedForwardProgramSlot::GeneralRouted(_)
                | FeedForwardProgramSlot::Parallel(_) => None,
            },
        ));
        let routed_layers = BindingLayers::of(target.blocks().iter().enumerate().filter_map(
            |(index, block)| match block.feed_forward()? {
                FeedForwardProgramSlot::Routed(binding) => {
                    Some((binding, sublayer_scope(index, 1)))
                }
                FeedForwardProgramSlot::Dense(_)
                | FeedForwardProgramSlot::GeneralRouted(_)
                | FeedForwardProgramSlot::Parallel(_) => None,
            },
        ));
        let general_routed_layers = BindingLayers::of(
            target
                .blocks()
                .iter()
                .enumerate()
                .filter_map(|(index, block)| match block.feed_forward()? {
                    FeedForwardProgramSlot::GeneralRouted(binding) => {
                        Some((binding, sublayer_scope(index, 1)))
                    }
                    FeedForwardProgramSlot::Dense(_)
                    | FeedForwardProgramSlot::Routed(_)
                    | FeedForwardProgramSlot::Parallel(_) => None,
                }),
        );
        let parallel_layers = BindingLayers::of(target.blocks().iter().enumerate().filter_map(
            |(index, block)| match block.feed_forward()? {
                FeedForwardProgramSlot::Parallel(binding) => {
                    Some((binding, sublayer_scope(index, 1)))
                }
                FeedForwardProgramSlot::Dense(_)
                | FeedForwardProgramSlot::Routed(_)
                | FeedForwardProgramSlot::GeneralRouted(_) => None,
            },
        ));
        for block in target.blocks() {
            match block.mixer() {
                MixerProgramSlot::Attention(b) if !self.target.attention.contains_key(&b) => {
                    if let Some(kernels) = self.attention(b, attention_layers.scopes(b))? {
                        self.target.attention.insert(b, kernels);
                    }
                }
                MixerProgramSlot::Recurrent(b) if !self.target.recurrent.contains_key(&b) => {
                    if let Some(kernels) = self.recurrent(b, recurrent_layers.scopes(b))? {
                        self.target.recurrent.insert(b, kernels);
                    }
                }
                MixerProgramSlot::StateSpace(b) if !self.target.state_space.contains_key(&b) => {
                    if let Some(kernels) = self.state_space(b, state_space_layers.scopes(b))? {
                        self.target.state_space.insert(b, kernels);
                    }
                }
                MixerProgramSlot::ShortConv(b) if !self.target.short_conv.contains_key(&b) => {
                    if let Some(kernels) = self.short_conv(b, short_conv_layers.scopes(b))? {
                        self.target.short_conv.insert(b, kernels);
                    }
                }
                MixerProgramSlot::Attention(_)
                | MixerProgramSlot::Recurrent(_)
                | MixerProgramSlot::StateSpace(_)
                | MixerProgramSlot::ShortConv(_) => {}
            }
            match block.feed_forward() {
                Some(FeedForwardProgramSlot::Dense(b)) if !self.target.dense.contains_key(&b) => {
                    if let Some(kernels) = self.dense(b, dense_layers.scopes(b))? {
                        self.target.dense.insert(b, kernels);
                    }
                }
                Some(FeedForwardProgramSlot::Routed(b)) if !self.target.routed.contains_key(&b) => {
                    if let Some(kernels) = self.routed(b, routed_layers.scopes(b))? {
                        self.target.routed.insert(b, kernels);
                    }
                }
                Some(FeedForwardProgramSlot::GeneralRouted(b))
                    if !self.target.general_routed.contains_key(&b) =>
                {
                    if let Some(kernels) =
                        self.general_routed(b, general_routed_layers.scopes(b))?
                    {
                        self.target.general_routed.insert(b, kernels);
                    }
                }
                Some(FeedForwardProgramSlot::Parallel(b))
                    if !self.target.parallel.contains_key(&b) =>
                {
                    if let Some(kernels) = self.parallel(b, parallel_layers.scopes(b))? {
                        self.target.parallel.insert(b, kernels);
                    }
                }
                Some(
                    FeedForwardProgramSlot::Dense(_)
                    | FeedForwardProgramSlot::Routed(_)
                    | FeedForwardProgramSlot::GeneralRouted(_)
                    | FeedForwardProgramSlot::Parallel(_),
                )
                | None => {}
            }
        }
        let per_layer_layers = BindingLayers::of(
            target
                .blocks()
                .iter()
                .enumerate()
                .filter_map(|(index, block)| Some((block.per_layer()?, sublayer_scope(index, 2)))),
        );
        for b in target.blocks().iter().filter_map(|block| block.per_layer()) {
            if !self.target.per_layer.contains_key(&b) {
                if let Some(kernels) = self.per_layer(b, per_layer_layers.scopes(b))? {
                    self.target.per_layer.insert(b, kernels);
                }
            }
        }
        if let Some(b) = target.per_layer() {
            if let Some(kernels) = self.per_layer_entry(b)? {
                self.target.per_layer_entry.insert(b, kernels);
            }
        }
        let b = target.readout();
        let bindings = format!("{b:?}");
        let features = self.features(&bindings, b.norm, b.activation)?;
        let head = match b.head {
            ReadoutHead::Packed { weight, .. } => {
                let head = self.spec.tuned(
                    &mut self.tuning,
                    &HeadRowsTuning {
                        norm: b.norm,
                        weight,
                        activation: b.activation,
                        epsilon: self.epsilon,
                        rows: None,
                    },
                )?;
                let selected = self.spec.tuned(
                    &mut self.tuning,
                    &SelectedRowsTuning {
                        norm: b.norm,
                        weight,
                        activation: b.activation,
                        epsilon: self.epsilon,
                    },
                )?;
                head.zip(selected)
                    .map(|(head, selected)| ReadoutHeadKernels::Packed { head, selected })
            }
            ReadoutHead::Progressive => self
                .progressive(b.norm, b.activation, None)?
                .map(ReadoutHeadKernels::Progressive),
        };
        if let (Some(features), Some(head)) = (features, head) {
            self.target
                .readout
                .insert(b, ReadoutKernels { features, head });
        }
        if let Some(b) = target.features() {
            if let Some(kernel) = self.features(&format!("{b:?}"), b.norm, b.activation)? {
                self.target.features.insert(b, kernel);
            }
        }
        if let Some(taps) = target.taps() {
            self.taps(taps)?;
        }
        Ok(())
    }

    /// A separate draft's target taps: the tap, the fusion projection of the
    /// taps (`project_rows` into F32) and the conditioning feature rows.
    fn taps(&mut self, taps: &crate::TapProgramPlan) -> Result<(), CatalogFailure> {
        let activation = taps.activation;
        let bindings = format!("A={}", activation.name());
        let tap = fixed!(
            self.spec,
            tap_rows,
            bindings,
            tap_rows::Elements { A: activation }
        );
        let fusion = self.spec.tuned(
            &mut self.tuning,
            &ProjectRowsTuning {
                weight: taps.fusion,
                activation,
                kind: WeightKind::DraftFusion,
                output: Element::f32(),
                scopes: vec![WeightScope::Draft],
            },
        )?;
        let features = fixed!(
            self.spec,
            feature_rows,
            bindings,
            feature_rows::Elements { A: activation }
        );
        if let (Some(tap), Some(fusion), Some(features)) = (tap, fusion, features) {
            self.target.taps = Some(TapKernels {
                tap,
                fusion,
                features,
            });
        }
        Ok(())
    }

    /// A progressive head's certified levels and full exact pass over its
    /// planes' leading `rows` (every row when `None`).
    fn progressive(
        &mut self,
        norm: Element,
        activation: Element,
        rows: Option<u64>,
    ) -> Result<Option<ProgressiveReadoutKernels>, CatalogFailure> {
        let tuning = ProgressiveTuning {
            norm,
            activation,
            epsilon: self.epsilon,
            certified_rows: crate::programs::graph::readout::certified_rows(self.spec.backend()),
            rows,
        };
        let top = self.spec.tuned(&mut self.tuning, &TopRowsTuning(tuning))?;
        let refine = self.spec.tuned(&mut self.tuning, &RefineRowsTuning(tuning))?;
        let exact = self.spec.tuned(&mut self.tuning, &ExactRowsTuning(tuning))?;
        let planes = self.spec.tuned(&mut self.tuning, &PlanesRowsTuning(tuning))?;
        Ok(match (top, refine, exact, planes) {
            (Some(top), Some(refine), Some(exact), Some(planes)) => Some(ProgressiveReadoutKernels {
                top,
                refine,
                exact,
                planes,
            }),
            _ => None,
        })
    }

    /// `readout_features_rows` at this model's width.
    fn features(
        &mut self,
        bindings: &str,
        norm: Element,
        activation: Element,
    ) -> Result<Option<NativeKernel<readout_features_rows::Entry>>, CatalogFailure> {
        self.spec.fixed(
            bindings,
            &[("D", self.hidden)],
            readout_features_rows::native_entry_with(readout_features_rows::Elements {
                NW: norm,
                A: activation,
            }),
        )
    }

    fn attention(
        &mut self,
        binding: AttentionBinding,
        scopes: Vec<WeightScope>,
    ) -> Result<Option<AttentionKernels>, CatalogFailure> {
        let AttentionBinding {
            shape,
            activation,
            output,
            ..
        } = binding;
        let project = self.spec.tuned(
            &mut self.tuning,
            &AttentionProjectTuning {
                binding,
                scopes: scopes.clone(),
                epsilon: self.epsilon,
            },
        )?;
        let mix = || AttentionMix {
            activation,
            shape,
            scopes: scopes.clone(),
            epsilon: self.epsilon,
            decode_rows: None,
            listed: false,
        };
        let history = match binding.history {
            KvCodec::Dense => {
                // As for K8/V4, multi-row launches (draft blocks,
                // verification) get their own tuning identity, so their
                // grouped form is not displaced by the single-row one.
                let split_decode = matches!(
                    self.spec.backend(),
                    seismic::BackendName::Metal | seismic::BackendName::Vulkan
                ) && self.limits.max_rows >= 2;
                let mut single = mix();
                if split_decode {
                    single.decode_rows = Some(1..2);
                }
                let decode = self
                    .spec
                    .tuned(&mut self.tuning, &AttentionDecodeTuning(single))?;
                let verify = if split_decode {
                    let mut selected = mix();
                    selected.decode_rows =
                        Some(2..crate::operators::attention::graph::DECODE_ROWS + 1);
                    self.spec
                        .tuned(&mut self.tuning, &AttentionDecodeTuning(selected))?
                } else {
                    None
                };
                let prefill = self
                    .spec
                    .tuned(&mut self.tuning, &AttentionPrefillTuning(mix()))?;
                decode
                    .zip(prefill)
                    .map(|(decode, prefill)| AttentionHistoryKernels::Dense {
                        decode,
                        verify,
                        prefill,
                    })
            }
            KvCodec::AffineK8V4 => {
                let split_decode = matches!(
                    self.spec.backend(),
                    seismic::BackendName::Metal | seismic::BackendName::Vulkan
                ) && self.limits.max_rows >= 2;
                let mut single = mix();
                if self.spec.backend() == seismic::BackendName::Vulkan && split_decode {
                    single.decode_rows = Some(1..2);
                }
                let decode = self
                    .spec
                    .tuned(&mut self.tuning, &AttentionDecodeK8V4Tuning(single))?;
                let specialized_m4 = split_decode
                    && self.spec.backend() == seismic::BackendName::Metal
                    && shape.group == 8
                    && shape.width == 256
                    && self.limits.max_rows >= 4;
                let verify = if split_decode {
                    let mut selected = mix();
                    selected.decode_rows = Some(
                        2..if specialized_m4 {
                            4
                        } else {
                            crate::operators::attention::graph::DECODE_ROWS + 1
                        },
                    );
                    self.spec
                        .tuned(&mut self.tuning, &AttentionDecodeK8V4Tuning(selected))?
                } else {
                    None
                };
                // The packed form reuses each K/V tile across four Verify
                // rows. Give that workload its own tuning identity so
                // shorter and wider rows cannot displace its form.
                let verify_four = if specialized_m4 {
                    let mut selected = mix();
                    selected.decode_rows = Some(4..5);
                    self.spec
                        .tuned(&mut self.tuning, &AttentionDecodeK8V4Tuning(selected))?
                } else {
                    None
                };
                let verify_eight = if specialized_m4 && self.limits.max_rows >= 5 {
                    let mut selected = mix();
                    selected.decode_rows =
                        Some(5..crate::operators::attention::graph::DECODE_ROWS + 1);
                    self.spec
                        .tuned(&mut self.tuning, &AttentionDecodeK8V4Tuning(selected))?
                } else {
                    None
                };
                // Where the entry's forms differ by it, a launch that lists
                // the history row tiles its rows see and one that does not
                // are two kernels with their own admissible forms.
                // (`StateResourcePlan::lists_history_tiles`, of the opened
                // device's own fact; graph preparation checks they agree.)
                let lists = self.spec.forms_tensor_operations()
                    && binding.history == KvCodec::AffineK8V4;
                let prefill = self
                    .spec
                    .tuned(&mut self.tuning, &AttentionPrefillK8V4Tuning(mix()))?;
                let prefill_listed = if lists {
                    let mut listed = mix();
                    listed.listed = true;
                    self.spec
                        .tuned(&mut self.tuning, &AttentionPrefillK8V4Tuning(listed))?
                } else {
                    None
                };
                decode
                    .zip(prefill)
                    .map(|(decode, prefill)| AttentionHistoryKernels::AffineK8V4 {
                        decode,
                        verify,
                        verify_four,
                        verify_eight,
                        prefill,
                        prefill_listed,
                    })
            }
            KvCodec::RotatedK4V4 => {
                return Err(CatalogFailure::Preparation {
                    entry: "attention_decode",
                    bindings: format!("{shape:?}"),
                    outcome: "the native path has no rotated K4/V4 history entries".into(),
                })
            }
        };
        let output = match binding.tail {
            SublayerTail::Residual => self
                .spec
                .tuned(
                    &mut self.tuning,
                    &AttentionOutputTuning {
                        output,
                        activation,
                        shape,
                        scopes,
                    },
                )?
                .map(SublayerOutput::Residual),
            SublayerTail::PostNorm { norm, .. } => self
                .post_norm(
                    WeightKind::AttentionOutput,
                    output,
                    activation,
                    norm,
                    scopes,
                )?
                .map(SublayerOutput::PostNorm),
        };
        Ok(match (project, history, output) {
            (Some(project), Some(history), Some(output)) => Some(AttentionKernels {
                project,
                history,
                output,
            }),
            _ => None,
        })
    }

    /// The state-space entries (`operators::state_space`): the projection and
    /// output reuse the attention projection entries at this geometry.
    fn state_space(
        &mut self,
        binding: StateSpaceBinding,
        scopes: Vec<WeightScope>,
    ) -> Result<Option<StateSpaceKernels>, CatalogFailure> {
        let project = self.spec.tuned(
            &mut self.tuning,
            &StateSpaceProjectTuning {
                binding,
                scopes: scopes.clone(),
                epsilon: self.epsilon,
            },
        )?;
        let state = || StateSpaceState {
            activation: binding.activation,
            shape: binding.shape,
            scopes: scopes.clone(),
        };
        let step = self
            .spec
            .tuned(&mut self.tuning, &StateSpaceStepTuning(state()))?;
        let chunk = self
            .spec
            .tuned(&mut self.tuning, &StateSpaceChunkTuning(state()))?;
        let shape = binding.shape;
        let gate = fixed!(
            self.spec,
            state_space_gate,
            format!("{binding:?}"),
            state_space_gate::Elements {
                A: binding.activation
            },
            statics
                & [
                    ("G", shape.norm_groups()),
                    ("U", shape.norm_heads),
                    ("P", shape.head_width)
                ]
        );
        let output = self
            .spec
            .tuned(&mut self.tuning, &StateSpaceOutputTuning { binding, scopes })?;
        Ok(match (project, step, chunk, gate, output) {
            (Some(project), Some(step), Some(chunk), Some(gate), Some(output)) => {
                Some(StateSpaceKernels {
                    project,
                    step,
                    chunk,
                    gate,
                    output,
                })
            }
            _ => None,
        })
    }

    /// The short-convolution entries (`operators::short_conv`): the output
    /// reuses the attention output entry at one head of `channels`.
    fn short_conv(
        &mut self,
        binding: ShortConvBinding,
        scopes: Vec<WeightScope>,
    ) -> Result<Option<ShortConvKernels>, CatalogFailure> {
        let project = self.spec.tuned(
            &mut self.tuning,
            &ShortConvProjectTuning {
                binding,
                scopes: scopes.clone(),
                epsilon: self.epsilon,
            },
        )?;
        let rows = fixed!(
            self.spec,
            short_conv_rows,
            format!("{binding:?}"),
            short_conv_rows::Elements {
                A: binding.activation
            }
        );
        let output = self
            .spec
            .tuned(&mut self.tuning, &ShortConvOutputTuning { binding, scopes })?;
        Ok(match (project, rows, output) {
            (Some(project), Some(rows), Some(output)) => Some(ShortConvKernels {
                project,
                rows,
                output,
            }),
            _ => None,
        })
    }

    fn recurrent(
        &mut self,
        b: RecurrentBinding,
        scopes: Vec<WeightScope>,
    ) -> Result<Option<RecurrentKernels>, CatalogFailure> {
        let shape = RecurrentShape {
            key_heads: b.key_heads,
            value_heads: b.value_heads,
            width: b.width,
            convolution_width: b.convolution_width,
            scopes,
        };
        let state = || RecurrentState {
            recurrent_norm: b.recurrent_norm,
            activation: b.activation,
            shape: shape.clone(),
            epsilon: self.epsilon,
        };
        let form =
            StepForm::of(self.spec.backend()).map_err(|outcome| CatalogFailure::Preparation {
                entry: "gated_delta_step_convolved",
                bindings: format!("{b:?}"),
                outcome,
            })?;
        let project_tuning = || RecurrentProjectTuning {
            norm: b.norm,
            qkv: b.qkv,
            gate: b.gate,
            alpha: b.alpha,
            beta: b.beta,
            activation: b.activation,
            shape: shape.clone(),
            epsilon: self.epsilon,
        };
        let step = match form {
            StepForm::Step => self
                .spec
                .tuned(&mut self.tuning, &RecurrentStepTuning(state()))?
                .map(RecurrentStepKernels::Step),
            StepForm::Convolved => {
                let project = self.spec.tuned(
                    &mut self.tuning,
                    &RecurrentProjectConvolvedTuning(project_tuning()),
                )?;
                let step = self
                    .spec
                    .tuned(&mut self.tuning, &RecurrentStepConvolvedTuning(state()))?;
                project
                    .zip(step)
                    .map(|(project, step)| RecurrentStepKernels::Convolved { project, step })
            }
        };
        let chunk = self
            .spec
            .tuned(&mut self.tuning, &RecurrentChunkTuning(state()))?;
        let project = self.spec.tuned(&mut self.tuning, &project_tuning())?;
        let output = self.spec.tuned(
            &mut self.tuning,
            &RecurrentOutputTuning {
                output: b.output,
                activation: b.activation,
                shape,
            },
        )?;
        Ok(match (project, step, chunk, output) {
            (Some(project), Some(step), Some(chunk), Some(output)) => Some(RecurrentKernels {
                project,
                step,
                chunk,
                output,
            }),
            _ => None,
        })
    }

    fn dense(
        &mut self,
        binding: DenseBinding,
        scopes: Vec<WeightScope>,
    ) -> Result<Option<DenseKernels>, CatalogFailure> {
        let DenseBinding {
            features: _,
            norm,
            gate,
            up,
            down,
            activation,
            tail,
            // The tunings read their scale ports' extents from the plan.
            scales: _,
        } = binding;
        let expand = self.spec.tuned(
            &mut self.tuning,
            &DenseExpandTuning {
                norm,
                gate,
                up,
                activation,
                gate_kind: WeightKind::DenseGate,
                up_kind: WeightKind::DenseUp,
                // SiLU: the admitted dense form.
                function: 0,
                scopes: scopes.clone(),
                epsilon: self.epsilon,
            },
        )?;
        let output = match tail {
            SublayerTail::Residual => self
                .spec
                .tuned(
                    &mut self.tuning,
                    &DenseOutputTuning {
                        down,
                        activation,
                        down_kind: WeightKind::DenseDown,
                        scopes,
                    },
                )?
                .map(SublayerOutput::Residual),
            SublayerTail::PostNorm { norm, .. } => self
                .post_norm(WeightKind::DenseDown, down, activation, norm, scopes)?
                .map(SublayerOutput::PostNorm),
        };
        Ok(match (expand, output) {
            (Some(expand), Some(output)) => Some(DenseKernels { expand, output }),
            _ => None,
        })
    }

    /// A per-layer input sublayer: its gate, then the post-norm tail of its
    /// projection back to the hidden width.
    fn per_layer(
        &mut self,
        binding: crate::PerLayerBinding,
        scopes: Vec<WeightScope>,
    ) -> Result<Option<PerLayerKernels>, CatalogFailure> {
        let SublayerTail::PostNorm { norm, .. } = binding.tail else {
            return Err(CatalogFailure::Preparation {
                entry: "per_layer_gate",
                bindings: format!("{binding:?}"),
                outcome: "a per-layer input sublayer ends in a post-norm tail".into(),
            });
        };
        let gate = self.spec.tuned(
            &mut self.tuning,
            &super::tuning::per_layer::PerLayerGateTuning {
                binding,
                scopes: scopes.clone(),
            },
        )?;
        let output = self.post_norm(
            WeightKind::PerLayerProjection,
            binding.projection,
            binding.activation,
            norm,
            scopes,
        )?;
        Ok(gate
            .zip(output)
            .map(|(gate, output)| PerLayerKernels { gate, output }))
    }

    /// The per-layer entry: the embedding's projection into F32 per-layer
    /// channels and their combination with the host-table rows.
    fn per_layer_entry(
        &mut self,
        binding: crate::PerLayerEntryBinding,
    ) -> Result<Option<PerLayerEntryKernels>, CatalogFailure> {
        let project = self.spec.tuned(
            &mut self.tuning,
            &ProjectRowsTuning {
                weight: binding.projection,
                activation: binding.activation,
                kind: WeightKind::PerLayerModelProjection,
                output: Element::f32(),
                scopes: vec![WeightScope::Target],
            },
        )?;
        let spec = &mut self.spec;
        let activation = binding
            .activation
            .dtype()
            .ok_or_else(|| CatalogFailure::Preparation {
                entry: "import_dense",
                bindings: format!("{binding:?}"),
                outcome: "the activation element is not dense".into(),
            })?;
        let round = fixed!(
            spec,
            import_dense,
            dense_binding_name(seismic::DType::F32, activation),
            import_dense::Elements {
                E: Element::f32(),
                U: binding.activation,
            }
        );
        let table = match (binding.table_source.dtype(), binding.table.dtype()) {
            (Some(source), Some(resident)) => fixed!(
                spec,
                import_dense,
                dense_binding_name(source, resident),
                import_dense::Elements {
                    E: binding.table_source,
                    U: binding.table,
                }
            )
            .map(TableConversion::Dense),
            _ => fixed!(
                spec,
                repack_weight,
                element_binding_name(binding.table_source, binding.table),
                repack_weight::Elements {
                    E: binding.table_source,
                    U: binding.table,
                }
            )
            .map(TableConversion::Repack),
        };
        let copy = fixed!(spec, conditioning_overlay, "fixed");
        let inputs = self.spec.fixed(
            &format!("TW={},NW={}", binding.table.name(), binding.norm.name()),
            &[("L", binding.layers), ("P", binding.width)],
            per_layer_inputs::native_entry_with(per_layer_inputs::Elements {
                TW: binding.table,
                NW: binding.norm,
            }),
        )?;
        Ok(match (round, project, table, inputs, copy) {
            (Some(round), Some(project), Some(table), Some(inputs), Some(copy)) => {
                Some(PerLayerEntryKernels {
                    round,
                    project,
                    table,
                    inputs,
                    copy,
                })
            }
            _ => None,
        })
    }

    /// A dense branch beside a routed branch; `scopes` are the parallel
    /// sublayers' scopes, whose branches hold the weights.
    fn parallel(
        &mut self,
        binding: crate::ParallelBinding,
        scopes: Vec<WeightScope>,
    ) -> Result<Option<ParallelKernels>, CatalogFailure> {
        let branch_scopes = |branch: usize| {
            scopes
                .iter()
                .map(|scope| match *scope {
                    WeightScope::TargetSublayer(sublayer) => {
                        Ok(crate::operators::parallel::DenseBesideRouted::scopes(sublayer)[branch])
                    }
                    other => Err(CatalogFailure::Preparation {
                        entry: "moe_tail",
                        bindings: format!("{binding:?}"),
                        outcome: format!("parallel branches in {other:?}"),
                    }),
                })
                .collect::<Result<Vec<_>, _>>()
        };
        let (dense_scopes, routed_scopes) = (branch_scopes(0)?, branch_scopes(1)?);
        let dense = binding.dense;
        let expand = self.spec.tuned(
            &mut self.tuning,
            &DenseExpandTuning {
                norm: dense.norm,
                gate: dense.gate,
                up: dense.up,
                activation: dense.activation,
                gate_kind: WeightKind::DenseGate,
                up_kind: WeightKind::DenseUp,
                function: 0,
                scopes: dense_scopes.clone(),
                epsilon: self.epsilon,
            },
        )?;
        let down = self.spec.tuned(
            &mut self.tuning,
            &ProjectRowsTuning {
                weight: dense.down,
                activation: dense.activation,
                kind: WeightKind::DenseDown,
                output: Element::f32(),
                scopes: dense_scopes,
            },
        )?;
        let routed = self.general_routed(binding.routed, routed_scopes)?;
        let tail = fixed!(
            self.spec,
            moe_tail,
            format!("NW={}", binding.norm.name()),
            moe_tail::Elements { NW: binding.norm }
        );
        Ok(match (expand, down, routed, tail) {
            (Some(expand), Some(down), Some(routed), Some(tail)) => Some(ParallelKernels {
                expand,
                down,
                routed,
                tail,
            }),
            _ => None,
        })
    }

    /// A post-norm tail: the output projection `kind` into F32 rows, then the
    /// row op that normalizes them into the residual.
    fn post_norm(
        &mut self,
        kind: WeightKind,
        weight: Element,
        activation: Element,
        norm: Element,
        scopes: Vec<WeightScope>,
    ) -> Result<Option<PostNormKernels>, CatalogFailure> {
        let project = self.spec.tuned(
            &mut self.tuning,
            &ProjectRowsTuning {
                weight,
                activation,
                kind,
                output: Element::f32(),
                scopes,
            },
        )?;
        let residual = fixed!(
            self.spec,
            post_norm_residual,
            format!("NW={}", norm.name()),
            post_norm_residual::Elements { NW: norm }
        );
        Ok(project
            .zip(residual)
            .map(|(project, residual)| PostNormKernels { project, residual }))
    }

    /// The general routed entries (`operators::routed`), with the shared
    /// expert and latent projections as dense entries over their own weight
    /// kinds.
    fn general_routed(
        &mut self,
        binding: GeneralRoutedBinding,
        scopes: Vec<WeightScope>,
    ) -> Result<Option<GeneralRoutedKernels>, CatalogFailure> {
        let shape = binding.shape;
        let activation = binding.activation;
        let epsilon = self.epsilon;
        let select = self.spec.tuned(
            &mut self.tuning,
            &RoutedSelectTuning {
                binding,
                scopes: scopes.clone(),
                epsilon,
            },
        )?;
        let decode = || RoutedExpandDecodeTuning {
            binding,
            scopes: scopes.clone(),
        };
        let tiles = || RoutedExpertTilesTuning {
            binding,
            scopes: scopes.clone(),
        };
        let experts = if shape.experts_expansion.gated {
            let decode = self
                .spec
                .tuned(&mut self.tuning, &RoutedGateUpTuning(decode()))?;
            let grouped = self
                .spec
                .tuned(&mut self.tuning, &RoutedGatedTilesTuning(tiles()))?;
            decode
                .zip(grouped)
                .map(|(decode, grouped)| ExpertKernels::Gated { decode, grouped })
        } else {
            let decode = self
                .spec
                .tuned(&mut self.tuning, &RoutedUpTuning(decode()))?;
            let grouped = self
                .spec
                .tuned(&mut self.tuning, &RoutedUpTilesTuning(tiles()))?;
            decode
                .zip(grouped)
                .map(|(decode, grouped)| ExpertKernels::Plain { decode, grouped })
        };
        let down = self.spec.tuned(
            &mut self.tuning,
            &RoutedDownTuning {
                binding,
                scopes: scopes.clone(),
            },
        )?;
        let group = self.spec.tuned(
            &mut self.tuning,
            &RoutedGroupTuning {
                shape: routing_shape(&binding),
                layers: scopes.len(),
            },
        )?;
        let scatter = self.spec.tuned(
            &mut self.tuning,
            &RoutedScatterTuning {
                binding,
                layers: scopes.len(),
            },
        )?;
        let shared = match (shape.shared, binding.shared) {
            (Some((_, expansion)), Some((gate, up, down))) => {
                let expansion = match gate {
                    Some(gate) => self
                        .spec
                        .tuned(
                            &mut self.tuning,
                            &DenseExpandTuning {
                                norm: binding.norm,
                                gate,
                                up,
                                activation,
                                gate_kind: WeightKind::SharedGate,
                                up_kind: WeightKind::SharedUp,
                                function: expansion.activation,
                                scopes: scopes.clone(),
                                epsilon,
                            },
                        )?
                        .map(DenseExpansionKernel::Gated),
                    None => self
                        .spec
                        .tuned(
                            &mut self.tuning,
                            &DenseUpTuning {
                                norm: binding.norm,
                                up,
                                activation,
                                up_kind: WeightKind::SharedUp,
                                function: expansion.activation,
                                scopes: scopes.clone(),
                                epsilon,
                            },
                        )?
                        .map(DenseExpansionKernel::Plain),
                };
                let output = self.spec.tuned(
                    &mut self.tuning,
                    &DenseOutputTuning {
                        down,
                        activation,
                        down_kind: WeightKind::SharedDown,
                        scopes: scopes.clone(),
                    },
                )?;
                Some(expansion.zip(output))
            }
            (None, None) => None,
            _ => {
                return Err(CatalogFailure::Preparation {
                    entry: "dense_output",
                    bindings: format!("{binding:?}"),
                    outcome: "the shared expert's shape and binding disagree".into(),
                })
            }
        };
        let latent = match binding.latent {
            Some((down, up)) => {
                let project = self.spec.tuned(
                    &mut self.tuning,
                    &ProjectRowsTuning {
                        weight: down,
                        activation,
                        kind: WeightKind::LatentDown,
                        output: activation,
                        scopes: scopes.clone(),
                    },
                )?;
                let output = self.spec.tuned(
                    &mut self.tuning,
                    &DenseOutputTuning {
                        down: up,
                        activation,
                        down_kind: WeightKind::LatentUp,
                        scopes,
                    },
                )?;
                Some(project.zip(output))
            }
            None => None,
        };
        // `None` anywhere is a tuning count or a missing implementation.
        Ok(match (select, experts, down, group, scatter) {
            (Some(select), Some(experts), Some(down), Some(group), Some(scatter)) => {
                let shared = match shared {
                    Some(Some(shared)) => Some(shared),
                    Some(None) => return Ok(None),
                    None => None,
                };
                let latent = match latent {
                    Some(Some(latent)) => Some(latent),
                    Some(None) => return Ok(None),
                    None => None,
                };
                Some(GeneralRoutedKernels {
                    select,
                    experts,
                    down,
                    group,
                    scatter,
                    shared,
                    latent,
                })
            }
            _ => None,
        })
    }

    fn routed(
        &mut self,
        b: RoutedBinding,
        scopes: Vec<WeightScope>,
    ) -> Result<Option<RoutedKernels>, CatalogFailure> {
        let shape = RoutedShape {
            hidden: b.hidden,
            experts: b.experts,
            selected: b.selected,
            features: b.features,
            shared: b.shared,
        };
        let form = DecodeForm::of(self.spec.backend()).map_err(|outcome| {
            CatalogFailure::Preparation {
                entry: "routed_route_shared",
                bindings: format!("{b:?}"),
                outcome,
            }
        })?;
        let route_tuning = RoutedRouteTuning {
            norm: b.norm,
            router: b.router,
            activation: b.activation,
            shape,
            scopes: scopes.clone(),
            epsilon: self.epsilon,
            decode: form == DecodeForm::Expand,
        };
        let route = self.spec.tuned(&mut self.tuning, &route_tuning)?;
        let group = self.spec.tuned(
            &mut self.tuning,
            &RoutedGroupTuning {
                shape,
                layers: scopes.len(),
            },
        )?;
        let decode = match form {
            DecodeForm::Expand => self
                .spec
                .tuned(
                    &mut self.tuning,
                    &RoutedExpandTuning {
                        expert_gate: b.expert_gate,
                        expert_up: b.expert_up,
                        shared_gate: b.shared_gate,
                        shared_up: b.shared_up,
                        activation: b.activation,
                        shape,
                        scopes: scopes.clone(),
                    },
                )?
                .map(RoutedDecodeKernels::Expand),
            DecodeForm::SharedRoute => {
                let route = self.spec.tuned(
                    &mut self.tuning,
                    &RoutedRouteSharedTuning {
                        route: route_tuning,
                        shared_gate: b.shared_gate,
                        shared_up: b.shared_up,
                    },
                )?;
                let choices = self.spec.tuned(
                    &mut self.tuning,
                    &RoutedChoicesTuning {
                        expert_gate: b.expert_gate,
                        expert_up: b.expert_up,
                        activation: b.activation,
                        shape,
                        scopes: scopes.clone(),
                    },
                )?;
                route
                    .zip(choices)
                    .map(|(route, choices)| RoutedDecodeKernels::SharedRoute { route, choices })
            }
        };
        let output = self.spec.tuned(
            &mut self.tuning,
            &RoutedOutputTuning {
                expert_down: b.expert_down,
                shared_down: b.shared_down,
                activation: b.activation,
                shape,
                scopes: scopes.clone(),
            },
        )?;
        let experts = self.spec.tuned(
            &mut self.tuning,
            &RoutedExpertsTuning {
                expert_gate: b.expert_gate,
                expert_up: b.expert_up,
                expert_down: b.expert_down,
                activation: b.activation,
                shape,
                scopes: scopes.clone(),
            },
        )?;
        let combine = self.spec.tuned(
            &mut self.tuning,
            &RoutedCombineTuning {
                shared_gate: b.shared_gate,
                shared_up: b.shared_up,
                shared_down: b.shared_down,
                activation: b.activation,
                shape,
                scopes,
            },
        )?;
        Ok(match (route, decode, output, group, experts, combine) {
            (
                Some(route),
                Some(decode),
                Some(output),
                Some(group),
                Some(experts),
                Some(combine),
            ) => Some(RoutedKernels {
                route,
                decode,
                output,
                group,
                experts,
                combine,
            }),
            _ => None,
        })
    }

    fn head(&mut self, plan: &ProgramPlan) -> Result<(), CatalogFailure> {
        let Some(head_plan) = plan.head() else {
            return Ok(());
        };
        let scoped = |scope: fn(usize) -> WeightScope| {
            BindingLayers::of(
                head_plan
                    .blocks()
                    .iter()
                    .enumerate()
                    .map(|(index, binding)| (*binding, scope(index))),
            )
        };
        let layers = scoped(head_scope);
        let attention_layers = scoped(|index| head_sublayer_scope(index, 0));
        let feed_forward_layers = scoped(|index| head_sublayer_scope(index, 1));
        for &b in head_plan.blocks() {
            if self
                .head
                .as_ref()
                .is_some_and(|head| head.input.contains_key(&b))
            {
                continue;
            }
            let bindings = format!("{b:?}");
            let input = self.spec.tuned(
                &mut self.tuning,
                &DraftRowsTuning {
                    embedding: b.embedding_table,
                    embedding_norm: b.embedding_norm,
                    hidden_norm: b.hidden_norm,
                    combine: b.combine,
                    activation: b.activation,
                    scopes: layers.scopes(b),
                    epsilon: self.epsilon,
                },
            )?;
            // The draft head's history is always dense (its binding says so).
            let attention = self.attention(b.attention, attention_layers.scopes(b))?;
            let feed_forward = match b.feed_forward {
                FeedForwardProgramSlot::Dense(binding) => self
                    .dense(binding, feed_forward_layers.scopes(b))?
                    .map(AttestedFeedForward::Dense),
                FeedForwardProgramSlot::Routed(binding) => self
                    .routed(binding, feed_forward_layers.scopes(b))?
                    .map(AttestedFeedForward::Routed),
                // `operators::admit` keeps draft heads on the fused form.
                FeedForwardProgramSlot::GeneralRouted(_) | FeedForwardProgramSlot::Parallel(_) => {
                    return Err(CatalogFailure::Preparation {
                        entry: "routed_select",
                        bindings: format!("{b:?}"),
                        outcome: "draft heads run the fused routed form only".into(),
                    })
                }
            };
            let features = self.features(&bindings, b.output_norm, b.activation)?;
            let logits = match b.projection {
                HeadProjection::Packed(weight) => self
                    .spec
                    .tuned(
                        &mut self.tuning,
                        &HeadLogitsTuning {
                            weight,
                            activation: b.activation,
                        },
                    )?
                    .map(HeadLogitsKernels::Packed),
                HeadProjection::Progressive => self
                    .progressive(
                        b.output_norm,
                        b.activation,
                        Some(draft_vocabulary(self.vocabulary)),
                    )?
                    .map(HeadLogitsKernels::Progressive),
            };
            let head = self
                .head
                .as_mut()
                .expect("a head plan creates the head group");
            if let (
                Some(input),
                Some(attention),
                Some(feed_forward),
                Some(features),
                Some(logits),
            ) = (input, attention, feed_forward, features, logits)
            {
                head.input.insert(b, input);
                head.attention.insert(b, attention);
                match feed_forward {
                    AttestedFeedForward::Dense(dense) => {
                        head.dense.insert(b, dense);
                    }
                    AttestedFeedForward::Routed(routed) => {
                        head.routed.insert(b, routed);
                    }
                    AttestedFeedForward::GeneralRouted(_) | AttestedFeedForward::Parallel(_) => {
                        return Err(CatalogFailure::Preparation {
                            entry: "routed_select",
                            bindings: format!("{b:?}"),
                            outcome: "draft heads run the fused routed form only".into(),
                        })
                    }
                }
                head.features.insert(b, features);
                head.logits.insert(b, logits);
            }
        }
        // Token selection over the draft vocabulary.
        let vocabulary = draft_vocabulary(self.vocabulary);
        let shape = self
            .spec
            .tuned(&mut self.tuning, &ShapeRowsTuning { vocabulary })?;
        let sample = self
            .spec
            .tuned(&mut self.tuning, &SampleRowsTuning { vocabulary })?;
        let head = self
            .head
            .as_mut()
            .expect("a head plan creates the head group");
        head.shape = shape;
        head.sample = sample;
        Ok(())
    }

    /// A separate draft: per distinct binding, its layers' attention (block
    /// and injection) and dense entries; the block embedding; the output
    /// norm and projection of the proposing rows; DSpark's chain entries.
    fn draft(&mut self, plan: &ProgramPlan) -> Result<(), CatalogFailure> {
        let Some(draft_plan) = plan.draft() else {
            return Ok(());
        };
        let draft_scope = |index: usize, sublayer: u32| {
            WeightScope::DraftSublayer(SublayerIndex {
                block: u32::try_from(index).expect("draft block count fits u32"),
                sublayer,
            })
        };
        let attention_layers = BindingLayers::of(draft_plan.blocks().iter().enumerate().flat_map(
            |(index, b)| {
                [
                    (b.attention, draft_scope(index, 0)),
                    (b.injection, draft_scope(index, 0)),
                ]
            },
        ));
        let dense_layers = BindingLayers::of(
            draft_plan
                .blocks()
                .iter()
                .enumerate()
                .map(|(index, b)| (b.feed_forward, draft_scope(index, 1))),
        );
        for b in draft_plan.blocks() {
            for attention in [b.attention, b.injection] {
                if self.draft_kernels().attention.contains_key(&attention) {
                    continue;
                }
                if let Some(kernels) =
                    self.attention(attention, attention_layers.scopes(attention))?
                {
                    self.draft_kernels().attention.insert(attention, kernels);
                }
            }
            if !self.draft_kernels().dense.contains_key(&b.feed_forward) {
                if let Some(kernels) =
                    self.dense(b.feed_forward, dense_layers.scopes(b.feed_forward))?
                {
                    self.draft_kernels().dense.insert(b.feed_forward, kernels);
                }
            }
        }
        let b = draft_plan.embedding();
        let embedding = self.spec.fixed(
            &format!("{b:?}"),
            &[("D", self.hidden)],
            embedding_rows::native_entry_with(embedding_rows::Elements {
                EW: b.table,
                A: b.activation,
            }),
        )?;
        let readout_vocabulary =
            draft_readout_vocabulary(draft_plan.markov().is_some(), self.vocabulary);
        let head = self.spec.tuned(
            &mut self.tuning,
            &HeadRowsTuning {
                norm: draft_plan.output_norm(),
                weight: draft_plan.projection(),
                activation: draft_plan.activation(),
                epsilon: self.epsilon,
                rows: Some(readout_vocabulary),
            },
        )?;
        let shape = self.spec.tuned(
            &mut self.tuning,
            &ShapeRowsTuning {
                vocabulary: readout_vocabulary,
            },
        )?;
        let sample = self.spec.tuned(
            &mut self.tuning,
            &SampleRowsTuning {
                vocabulary: readout_vocabulary,
            },
        )?;
        let widen = fixed!(
            self.spec,
            widen_rows,
            format!("A={}", draft_plan.activation().name()),
            widen_rows::Elements {
                A: draft_plan.activation()
            }
        );
        let markov = match draft_plan.markov() {
            None => None,
            Some(markov) => {
                let activation = draft_plan.activation();
                let bindings = format!("{markov:?}");
                let rank = markov.rank;
                let embedding = self.spec.fixed(
                    &bindings,
                    &[("D", rank)],
                    embedding_rows::native_entry_with(embedding_rows::Elements {
                        EW: markov.embedding,
                        A: activation,
                    }),
                )?;
                let projection = self.spec.tuned(
                    &mut self.tuning,
                    &DenseOutputTuning {
                        down: markov.projection,
                        activation,
                        down_kind: WeightKind::MarkovProjection,
                        scopes: vec![WeightScope::Draft],
                    },
                )?;
                let features = self.features(&bindings, draft_plan.output_norm(), activation)?;
                let confidence = fixed!(
                    self.spec,
                    draft_confidence,
                    bindings,
                    draft_confidence::Elements { A: activation },
                    statics & [("D", self.hidden), ("R", rank)]
                );
                match (embedding, projection, features, confidence) {
                    (Some(embedding), Some(projection), Some(features), Some(confidence)) => {
                        Some(MarkovKernels {
                            embedding,
                            projection,
                            features,
                            confidence,
                        })
                    }
                    _ => None,
                }
            }
        };
        let dflash2 = match draft_plan.dflash2() {
            None => None,
            Some(binding) => self.dflash2(draft_plan, binding)?,
        };
        let draft = self.draft_kernels();
        draft.embedding = embedding;
        draft.head = head;
        draft.shape = shape;
        draft.sample = sample;
        draft.widen = widen;
        draft.markov = markov;
        draft.dflash2 = dflash2;
        Ok(())
    }

    /// DFlash2's block pass and candidate path: the layer and output norms
    /// over rows, every unfused projection (per kind, weight and published
    /// element, over every layer sharing it), the two convolution halves,
    /// the gated product, the top-k, both codebook gathers and the path
    /// step. `None` during a tuning count.
    fn dflash2(
        &mut self,
        plan: &crate::DraftProgramPlan,
        binding: &crate::Dflash2Binding,
    ) -> Result<Option<Dflash2Kernels>, CatalogFailure> {
        let activation = plan.activation();
        let selector = binding.selector;
        let (projections, norm_elements) = super::draft::dflash2_entries(plan, binding);
        let mut prepared = Dflash2Projections::new();
        let mut complete = true;
        for ((kind, weight, output), scopes) in projections {
            match self.spec.tuned(
                &mut self.tuning,
                &ProjectRowsTuning {
                    weight,
                    activation,
                    kind,
                    output,
                    scopes,
                },
            )? {
                Some(kernel) => {
                    prepared.insert((kind, weight, output), kernel);
                }
                None => complete = false,
            }
        }
        let mut norms = HashMap::new();
        for norm in norm_elements {
            match self.features(&format!("dflash2 NW={}", norm.name()), norm, activation)? {
                Some(kernel) => {
                    norms.insert(norm, kernel);
                }
                None => complete = false,
            }
        }
        let bindings = format!("A={}", activation.name());
        let convolve_input = fixed!(
            self.spec,
            draft_convolve_input,
            bindings,
            draft_convolve_input::Elements { A: activation }
        );
        let convolve_residual = fixed!(self.spec, draft_convolve_residual, "");
        let gated = fixed!(
            self.spec,
            draft_gated_rows,
            bindings,
            draft_gated_rows::Elements { A: activation }
        );
        let top_k = fixed!(self.spec, draft_top_k, "");
        let path = fixed!(
            self.spec,
            draft_path_step,
            bindings,
            draft_path_step::Elements { A: activation }
        );
        let codebook = |spec: &mut Specializer, table: Element| {
            spec.fixed(
                &format!("dflash2 EW={}", table.name()),
                &[("D", selector.rank)],
                embedding_rows::native_entry_with(embedding_rows::Elements {
                    EW: table,
                    A: activation,
                }),
            )
        };
        let predecessor = codebook(&mut self.spec, selector.predecessor)?;
        let successor = codebook(&mut self.spec, selector.successor)?;
        Ok(
            match (
                complete,
                convolve_input,
                convolve_residual,
                gated,
                top_k,
                path,
                predecessor,
                successor,
            ) {
                (
                    true,
                    Some(convolve_input),
                    Some(convolve_residual),
                    Some(gated),
                    Some(top_k),
                    Some(path),
                    Some(predecessor),
                    Some(successor),
                ) => Some(Dflash2Kernels {
                    norms,
                    projections: prepared,
                    convolve_input,
                    convolve_residual,
                    gated,
                    top_k,
                    predecessor,
                    successor,
                    path,
                }),
                _ => None,
            },
        )
    }

    fn draft_kernels(&mut self) -> &mut DraftKernels {
        self.draft
            .as_mut()
            .expect("a draft plan creates the draft group")
    }

    fn vision(&mut self, plan: &ProgramPlan) -> Result<(), CatalogFailure> {
        let Some(vision_plan) = plan.vision() else {
            return Ok(());
        };
        let spec = &mut self.spec;
        let vision = self
            .vision
            .as_mut()
            .expect("a vision plan creates the vision group");
        for kernel in vision_plan.kernels() {
            let label = format!("{kernel:?}");
            let statics = kernel.statics.as_slice();
            macro_rules! element {
                ($module:ident, $name:literal) => {
                    super::vision::element(<$module::Entry as seismic::Entry>::NAME, kernel, $name)?
                };
            }
            macro_rules! prepare {
                ($module:ident, $map:ident, $elements:expr) => {{
                    let elements = $elements;
                    if let Some(native) =
                        fixed!(spec, $module, label, elements, statics & statics)
                    {
                        vision.$map.insert(kernel.clone(), native);
                    }
                }};
            }
            match kernel.entry {
                VisionEntry::PatchStem => prepare!(
                    vision_patch_stem,
                    patch_stem,
                    vision_patch_stem::Elements {
                        W0: element!(vision_patch_stem, "W0"),
                        W1: element!(vision_patch_stem, "W1"),
                        B: element!(vision_patch_stem, "B"),
                        PE: element!(vision_patch_stem, "PE"),
                    }
                ),
                VisionEntry::Norm => prepare!(
                    vision_norm,
                    norm,
                    vision_norm::Elements {
                        NWE: element!(vision_norm, "NWE"),
                        NBE: element!(vision_norm, "NBE"),
                        Y: element!(vision_norm, "Y"),
                    }
                ),
                VisionEntry::Linear => prepare!(
                    vision_linear,
                    linear,
                    vision_linear::Elements {
                        A: element!(vision_linear, "A"),
                        W: element!(vision_linear, "W"),
                        B: element!(vision_linear, "B"),
                        Y: element!(vision_linear, "Y"),
                    }
                ),
                VisionEntry::Clamp => prepare!(
                    vision_clamp,
                    clamp,
                    vision_clamp::Elements {
                        A: element!(vision_clamp, "A"),
                    }
                ),
                VisionEntry::Attention => prepare!(
                    vision_attention,
                    attention,
                    vision_attention::Elements {
                        A: element!(vision_attention, "A"),
                    }
                ),
                VisionEntry::Pool => prepare!(
                    vision_pool,
                    pool,
                    vision_pool::Elements {
                        SB: element!(vision_pool, "SB"),
                        SS: element!(vision_pool, "SS"),
                    }
                ),
                VisionEntry::Position => prepare!(
                    vision_position,
                    position,
                    vision_position::Elements {
                        PE: element!(vision_position, "PE"),
                    }
                ),
                VisionEntry::PostNormResidual => prepare!(
                    post_norm_residual,
                    post_norm,
                    post_norm_residual::Elements {
                        NW: element!(post_norm_residual, "NW"),
                    }
                ),
            }
        }
        Ok(())
    }
}
