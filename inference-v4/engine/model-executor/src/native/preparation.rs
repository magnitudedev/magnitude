use super::specialization::Specializer;
use super::tuning::{
    attention::{
        AttentionDecodeK8V4Tuning, AttentionDecodeTuning, AttentionMix, AttentionOutputTuning,
        AttentionPrefillK8V4Tuning, AttentionPrefillTuning, AttentionProjectTuning,
    },
    cases::{DenseExpandTuning, DenseOutputTuning},
    readout::{
        HeadLogitsTuning, HeadRowsTuning, SampleRowsTuning, SelectedRowsTuning, ShapeRowsTuning,
    },
    recurrent::{
        RecurrentChunkTuning, RecurrentOutputTuning, RecurrentProjectTuning, RecurrentShape,
        RecurrentState, RecurrentStepTuning,
    },
    routed::{
        RoutedCombineTuning, RoutedExpandTuning, RoutedExpertsTuning, RoutedGroupTuning,
        RoutedOutputTuning, RoutedRouteTuning, RoutedShape,
    },
    TunedEntry, Tuner, TuningContext, TuningLimits, TuningWeights,
};
use super::*;
use crate::ModelLoadPlan;
use magnitude_model_contracts::WeightScope;
use magnitude_model_state::KvCodec;

/// The phase-one catalog. Every handle is prepared before qualification and
/// retained for warm calls; this type has no API capable of preparing again.
#[derive(Debug)]
pub(super) struct NativePreparationCache {
    pub(super) owner: Tensor,
    pub(super) import: ImportKernels,
    pub(super) glue: GlueKernels,
    pub(super) target: TargetKernels,
    pub(super) head: Option<HeadKernels>,
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
/// when the backend has no implementation, which the specializer records.
macro_rules! fixed {
    ($spec:expr, $device:expr, $module:ident, $bindings:expr, $elements:expr) => {
        fixed!($spec, $device, $module, $bindings, $elements, statics &[])
    };
    ($spec:expr, $device:expr, $module:ident, $bindings:expr, $elements:expr, statics $statics:expr) => {
        $spec.fixed::<$module::Entry>(&$bindings, $statics, |specialization| {
            $module::native_for_device_with($device, $elements, specialization)
        })?
    };
    ($spec:expr, $device:expr, $module:ident, $bindings:expr) => {
        $spec.fixed::<$module::Entry>(&$bindings, &[], |specialization| {
            $module::native_for_device($device, specialization)
        })?
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

fn block_scope(index: usize) -> WeightScope {
    WeightScope::TargetBlock(u32::try_from(index).expect("block count fits u32"))
}

fn head_scope(index: usize) -> WeightScope {
    WeightScope::HeadBlock(u32::try_from(index).expect("head block count fits u32"))
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
                        device,
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
                        device,
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
        let owner = Tensor::zeros(device, Element::u32(), &[1]).map_err(|error| {
            CatalogFailure::Preparation {
                entry: "catalog_owner",
                bindings: "A=u32".into(),
                outcome: error.to_string(),
            }
        })?;
        // A census walk counts the program's tuning units, so the model's
        // budget can be shared before anything is tuned.
        let mut census = Preparation::new(
            device,
            plan,
            tuning,
            Specializer::census(device),
            Tuner::census(device, tuning, limits, TuningWeights::new(device, load, tuning.weights, &import)),
        );
        census.walk(plan)?;
        let budgets = census.tuner.budgets();
        let weights = TuningWeights::new(device, load, tuning.weights, &import);
        let mut preparation = Preparation::new(
            device,
            plan,
            tuning,
            spec,
            Tuner::new(device, tuning, limits, weights, budgets),
        );
        let glue = preparation.walk(plan)?;
        let Preparation {
            tuner,
            target,
            head,
            vision,
            ..
        } = preparation;
        let tuned = tuner.tuned();
        Ok(Self {
            owner,
            import,
            glue,
            target,
            head,
            vision,
            tuned,
        })
    }
}

struct Preparation<'a> {
    device: &'a Device,
    spec: Specializer<'a>,
    tuner: Tuner<'a>,
    epsilon: f32,
    /// Model width: the static dimension of the feature readout.
    hidden: u64,
    /// The target's vocabulary: the static dimension of token selection.
    vocabulary: u64,
    vision_statics: Option<super::vision::VisionStatics>,
    target: TargetKernels,
    head: Option<HeadKernels>,
    vision: Option<VisionKernels>,
}

impl<'a> Preparation<'a> {
    fn new(
        device: &'a Device,
        plan: &ProgramPlan,
        tuning: TuningContext<'a>,
        spec: Specializer<'a>,
        tuner: Tuner<'a>,
    ) -> Self {
        Self {
            device,
            spec,
            tuner,
            epsilon: tuning.definition.geometry.epsilon as f32,
            hidden: tuning.definition.geometry.hidden,
            vocabulary: tuning.definition.geometry.vocabulary,
            vision_statics: tuning
                .definition
                .vision
                .as_ref()
                .map(|vision| super::vision::VisionStatics::of(&vision.geometry)),
            target: TargetKernels::default(),
            head: plan.head().map(|_| HeadKernels::default()),
            vision: plan.vision().map(|_| VisionKernels::default()),
        }
    }

    /// Prepare every program entry, in a fixed order.
    fn walk(&mut self, plan: &ProgramPlan) -> Result<GlueKernels, CatalogFailure> {
        let glue = self.glue(plan)?;
        self.target(plan)?;
        self.head(plan)?;
        self.vision(plan)?;
        Ok(glue)
    }

    /// Token selection, the conditioning overlay and the state row copies.
    fn glue(&mut self, plan: &ProgramPlan) -> Result<GlueKernels, CatalogFailure> {
        let device = self.device;
        let copies = plan.state().copies();
        let copy = |spec: &mut Specializer<'_>, element: Element, bindings: &'static str| {
            if !copies.contains(&element) {
                return Ok(None);
            }
            Ok::<_, CatalogFailure>(fixed!(
                spec,
                device,
                copy_rows,
                bindings,
                copy_rows::Elements { A: element }
            ))
        };
        Ok(GlueKernels {
            shape_rows: self.spec.tuned(
                &mut self.tuner,
                &ShapeRowsTuning {
                    vocabulary: self.vocabulary,
                },
            )?,
            sample_rows: self.spec.tuned(
                &mut self.tuner,
                &SampleRowsTuning {
                    vocabulary: self.vocabulary,
                },
            )?,
            conditioning_overlay: fixed!(self.spec, device, conditioning_overlay, "fixed"),
            copy_rows_f32: copy(&mut self.spec, Element::f32(), "A=f32")?,
            copy_rows_f16: copy(&mut self.spec, Element::f16(), "A=f16")?,
            copy_rows_bf16: copy(&mut self.spec, Element::bf16(), "A=bf16")?,
            copy_rows_u32: copy(&mut self.spec, Element::u32(), "A=u32")?,
        })
    }

    fn target(&mut self, plan: &ProgramPlan) -> Result<(), CatalogFailure> {
        let device = self.device;
        let spec = &mut self.spec;
        let target = plan.target();
        let b = target.embedding();
        if let Some(kernel) = spec.fixed::<embedding_rows::Entry>(
            &format!("{b:?}"),
            &[("D", self.hidden)],
            |specialization| {
                embedding_rows::native_for_device_with(
                    device,
                    embedding_rows::Elements {
                        EW: b.table,
                        A: b.activation,
                    },
                    specialization,
                )
            },
        )? {
            self.target.embedding.insert(b, kernel);
        }
        let attention_layers = BindingLayers::of(target.blocks().iter().enumerate().filter_map(
            |(index, block)| match block.mixer() {
                MixerProgramSlot::Attention(binding) => Some((binding, block_scope(index))),
                MixerProgramSlot::Recurrent(_) => None,
            },
        ));
        let recurrent_layers = BindingLayers::of(target.blocks().iter().enumerate().filter_map(
            |(index, block)| match block.mixer() {
                MixerProgramSlot::Recurrent(binding) => Some((binding, block_scope(index))),
                MixerProgramSlot::Attention(_) => None,
            },
        ));
        let dense_layers = BindingLayers::of(target.blocks().iter().enumerate().filter_map(
            |(index, block)| match block.feed_forward() {
                FeedForwardProgramSlot::Dense(binding) => Some((binding, block_scope(index))),
                FeedForwardProgramSlot::Routed(_) => None,
            },
        ));
        let routed_layers = BindingLayers::of(target.blocks().iter().enumerate().filter_map(
            |(index, block)| match block.feed_forward() {
                FeedForwardProgramSlot::Routed(binding) => Some((binding, block_scope(index))),
                FeedForwardProgramSlot::Dense(_) => None,
            },
        ));
        for block in target.blocks() {
            match block.mixer() {
                MixerProgramSlot::Attention(b) if !self.target.attention.contains_key(&b) => {
                    if let Some(kernels) = self.attention(
                        b.shape,
                        b.norm,
                        b.query_gate,
                        b.key,
                        b.value,
                        b.output,
                        b.activation,
                        b.history,
                        attention_layers.scopes(b),
                    )? {
                        self.target.attention.insert(b, kernels);
                    }
                }
                MixerProgramSlot::Recurrent(b) if !self.target.recurrent.contains_key(&b) => {
                    if let Some(kernels) = self.recurrent(b, recurrent_layers.scopes(b))? {
                        self.target.recurrent.insert(b, kernels);
                    }
                }
                MixerProgramSlot::Attention(_) | MixerProgramSlot::Recurrent(_) => {}
            }
            match block.feed_forward() {
                FeedForwardProgramSlot::Dense(b) if !self.target.dense.contains_key(&b) => {
                    if let Some(kernels) = self.dense(
                        b.norm,
                        b.gate,
                        b.up,
                        b.down,
                        b.activation,
                        dense_layers.scopes(b),
                    )? {
                        self.target.dense.insert(b, kernels);
                    }
                }
                FeedForwardProgramSlot::Routed(b) if !self.target.routed.contains_key(&b) => {
                    if let Some(kernels) = self.routed(b, routed_layers.scopes(b))? {
                        self.target.routed.insert(b, kernels);
                    }
                }
                FeedForwardProgramSlot::Dense(_) | FeedForwardProgramSlot::Routed(_) => {}
            }
        }
        let b = target.readout();
        let bindings = format!("{b:?}");
        let features = self.features(&bindings, b.norm, b.activation)?;
        let head = self.spec.tuned(
            &mut self.tuner,
            &HeadRowsTuning {
                norm: b.norm,
                weight: b.weight,
                activation: b.activation,
                epsilon: self.epsilon,
            },
        )?;
        let selected = self.spec.tuned(
            &mut self.tuner,
            &SelectedRowsTuning {
                norm: b.norm,
                weight: b.weight,
                activation: b.activation,
                epsilon: self.epsilon,
            },
        )?;
        if let (Some(features), Some(head)) = (features, head) {
            self.target
                .readout
                .insert(b, ReadoutKernels { features, head });
        }
        if let Some(selected) = selected {
            self.target.selected.insert(b, selected);
        }
        if let Some(b) = target.features() {
            if let Some(kernel) = self.features(&format!("{b:?}"), b.norm, b.activation)? {
                self.target.features.insert(b, kernel);
            }
        }
        Ok(())
    }

    /// `readout_features_rows` at this model's width.
    fn features(
        &mut self,
        bindings: &str,
        norm: Element,
        activation: Element,
    ) -> Result<Option<NativeKernel<readout_features_rows::Entry>>, CatalogFailure> {
        let device = self.device;
        self.spec.fixed::<readout_features_rows::Entry>(
            bindings,
            &[("D", self.hidden)],
            |specialization| {
                readout_features_rows::native_for_device_with(
                    device,
                    readout_features_rows::Elements {
                        NW: norm,
                        A: activation,
                    },
                    specialization,
                )
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &mut self,
        shape: AttentionShape,
        norm: Element,
        query_gate: Element,
        key: Element,
        value: Element,
        output: Element,
        activation: Element,
        history: KvCodec,
        scopes: Vec<WeightScope>,
    ) -> Result<Option<AttentionKernels>, CatalogFailure> {
        let project = self.spec.tuned(
            &mut self.tuner,
            &AttentionProjectTuning {
                norm,
                query_gate,
                key,
                value,
                activation,
                shape,
                scopes: scopes.clone(),
                epsilon: self.epsilon,
            },
        )?;
        let mix = || AttentionMix {
            activation,
            shape,
            scopes: scopes.clone(),
            epsilon: self.epsilon,
        };
        let history = match history {
            KvCodec::Dense => {
                let decode = self
                    .spec
                    .tuned(&mut self.tuner, &AttentionDecodeTuning(mix()))?;
                let prefill = self
                    .spec
                    .tuned(&mut self.tuner, &AttentionPrefillTuning(mix()))?;
                decode
                    .zip(prefill)
                    .map(|(decode, prefill)| AttentionHistoryKernels::Dense { decode, prefill })
            }
            KvCodec::AffineK8V4 => {
                let decode = self
                    .spec
                    .tuned(&mut self.tuner, &AttentionDecodeK8V4Tuning(mix()))?;
                let prefill = self
                    .spec
                    .tuned(&mut self.tuner, &AttentionPrefillK8V4Tuning(mix()))?;
                decode.zip(prefill).map(|(decode, prefill)| {
                    AttentionHistoryKernels::AffineK8V4 { decode, prefill }
                })
            }
            KvCodec::RotatedK4V4 => {
                return Err(CatalogFailure::Preparation {
                    entry: "gated_attention_decode",
                    bindings: format!("{shape:?}"),
                    outcome: "the native path has no rotated K4/V4 history entries".into(),
                })
            }
        };
        let output = self.spec.tuned(
            &mut self.tuner,
            &AttentionOutputTuning {
                output,
                activation,
                shape,
                scopes,
            },
        )?;
        Ok(match (project, history, output) {
            (Some(project), Some(history), Some(output)) => Some(AttentionKernels {
                project,
                history,
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
            activation: b.activation,
            shape: shape.clone(),
            epsilon: self.epsilon,
        };
        let step = self
            .spec
            .tuned(&mut self.tuner, &RecurrentStepTuning(state()))?;
        let chunk = self
            .spec
            .tuned(&mut self.tuner, &RecurrentChunkTuning(state()))?;
        let project = self.spec.tuned(
            &mut self.tuner,
            &RecurrentProjectTuning {
                norm: b.norm,
                qkv: b.qkv,
                gate: b.gate,
                alpha: b.alpha,
                beta: b.beta,
                activation: b.activation,
                shape: shape.clone(),
                epsilon: self.epsilon,
            },
        )?;
        let output = self.spec.tuned(
            &mut self.tuner,
            &RecurrentOutputTuning {
                recurrent_norm: b.recurrent_norm,
                output: b.output,
                activation: b.activation,
                shape,
                epsilon: self.epsilon,
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
        norm: Element,
        gate: Element,
        up: Element,
        down: Element,
        activation: Element,
        scopes: Vec<WeightScope>,
    ) -> Result<Option<DenseKernels>, CatalogFailure> {
        let expand = self.spec.tuned(
            &mut self.tuner,
            &DenseExpandTuning {
                norm,
                gate,
                up,
                activation,
                scopes: scopes.clone(),
                epsilon: self.epsilon,
            },
        )?;
        let output = self.spec.tuned(
            &mut self.tuner,
            &DenseOutputTuning {
                down,
                activation,
                scopes,
            },
        )?;
        Ok(match (expand, output) {
            (Some(expand), Some(output)) => Some(DenseKernels { expand, output }),
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
        let route = self.spec.tuned(
            &mut self.tuner,
            &RoutedRouteTuning {
                norm: b.norm,
                router: b.router,
                activation: b.activation,
                shape,
                scopes: scopes.clone(),
                epsilon: self.epsilon,
            },
        )?;
        let group = self
            .spec
            .tuned(&mut self.tuner, &RoutedGroupTuning { shape })?;
        let expand = self.spec.tuned(
            &mut self.tuner,
            &RoutedExpandTuning {
                expert_gate: b.expert_gate,
                expert_up: b.expert_up,
                shared_gate: b.shared_gate,
                shared_up: b.shared_up,
                activation: b.activation,
                shape,
                scopes: scopes.clone(),
            },
        )?;
        let output = self.spec.tuned(
            &mut self.tuner,
            &RoutedOutputTuning {
                expert_down: b.expert_down,
                shared_down: b.shared_down,
                activation: b.activation,
                shape,
                scopes: scopes.clone(),
            },
        )?;
        let experts = self.spec.tuned(
            &mut self.tuner,
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
            &mut self.tuner,
            &RoutedCombineTuning {
                shared_gate: b.shared_gate,
                shared_up: b.shared_up,
                shared_down: b.shared_down,
                activation: b.activation,
                shape,
                scopes,
            },
        )?;
        Ok(match (route, expand, output, group, experts, combine) {
            (Some(route), Some(expand), Some(output), Some(group), Some(experts), Some(combine)) => {
                Some(RoutedKernels {
                    route,
                    expand,
                    output,
                    group,
                    experts,
                    combine,
                })
            }
            _ => None,
        })
    }

    fn head(&mut self, plan: &ProgramPlan) -> Result<(), CatalogFailure> {
        let Some(head_plan) = plan.head() else {
            return Ok(());
        };
        let layers = BindingLayers::of(
            head_plan
                .blocks()
                .iter()
                .enumerate()
                .map(|(index, binding)| (*binding, head_scope(index))),
        );
        let device = self.device;
        for &b in head_plan.blocks() {
            if self
                .head
                .as_ref()
                .is_some_and(|head| head.input.contains_key(&b))
            {
                continue;
            }
            let bindings = format!("{b:?}");
            let spec = &mut self.spec;
            let input = fixed!(
                spec,
                device,
                draft_rows,
                bindings,
                draft_rows::Elements {
                    EW: b.embedding_table,
                    A: b.activation,
                    EN: b.embedding_norm,
                    HN: b.hidden_norm,
                    CW: b.combine,
                },
                statics &[("D", b.attention_shape.hidden)]
            );
            let attention = self.attention(
                b.attention_shape,
                b.input_norm,
                b.query_gate,
                b.key,
                b.value,
                b.attention_output,
                b.activation,
                // The draft head's history is always dense.
                KvCodec::Dense,
                layers.scopes(b),
            )?;
            let dense = self.dense(
                b.feedforward_norm,
                b.gate,
                b.up,
                b.down,
                b.activation,
                layers.scopes(b),
            )?;
            let features = self.features(&bindings, b.output_norm, b.activation)?;
            let logits = self.spec.tuned(
                &mut self.tuner,
                &HeadLogitsTuning {
                    weight: b.projection,
                    activation: b.activation,
                },
            )?;
            let head = self
                .head
                .as_mut()
                .expect("a head plan creates the head group");
            if let (Some(input), Some(attention), Some(dense), Some(features), Some(logits)) =
                (input, attention, dense, features, logits)
            {
                head.input.insert(b, input);
                head.attention.insert(b, attention);
                head.dense.insert(b, dense);
                head.features.insert(b, features);
                head.logits.insert(b, logits);
            }
        }
        // Token selection over the draft vocabulary.
        let vocabulary = draft_vocabulary(self.vocabulary);
        let shape = self
            .spec
            .tuned(&mut self.tuner, &ShapeRowsTuning { vocabulary })?;
        let sample = self
            .spec
            .tuned(&mut self.tuner, &SampleRowsTuning { vocabulary })?;
        let head = self
            .head
            .as_mut()
            .expect("a head plan creates the head group");
        head.shape = shape;
        head.sample = sample;
        Ok(())
    }

    fn vision(&mut self, plan: &ProgramPlan) -> Result<(), CatalogFailure> {
        let Some(vision_plan) = plan.vision() else {
            return Ok(());
        };
        let (device, spec) = (self.device, &mut self.spec);
        let statics = self
            .vision_statics
            .expect("a vision plan has a vision definition");
        let vision = self
            .vision
            .as_mut()
            .expect("a vision plan creates the vision group");
        let b = vision_plan.patch();
        if let Some(kernel) = fixed!(
            spec,
            device,
            qwen_vision_stem,
            format!("{b:?}"),
            qwen_vision_stem::Elements {
                W0: b.temporal_weight_0,
                W1: b.temporal_weight_1,
                B: b.bias,
                PE: b.position,
            },
            statics &statics.stem
        ) {
            vision.stem.insert(b, kernel);
        }
        for &b in vision_plan.blocks() {
            if vision.blocks.contains_key(&b) {
                continue;
            }
            if let Some(kernel) = fixed!(
                spec,
                device,
                qwen_vision_block,
                format!("{b:?}"),
                qwen_vision_block::Elements {
                    A: b.activation,
                    N1W: b.input_norm_weight,
                    N1B: b.input_norm_bias,
                    QW: b.qkv_weight,
                    QB: b.qkv_bias,
                    PW: b.attention_output,
                    PB: b.attention_output_bias,
                    N2W: b.feedforward_norm_weight,
                    N2B: b.feedforward_norm_bias,
                    UW: b.up,
                    UB: b.up_bias,
                    DW: b.down,
                    DB: b.down_bias,
                },
                statics &statics.block
            ) {
                vision.blocks.insert(b, kernel);
            }
        }
        let b = vision_plan.merger();
        if let Some(kernel) = fixed!(
            spec,
            device,
            qwen_vision_merger,
            format!("{b:?}"),
            qwen_vision_merger::Elements {
                A: b.activation,
                NW: b.output_norm_weight,
                NB: b.output_norm_bias,
                UW: b.hidden,
                UB: b.hidden_bias,
                DW: b.output,
                DB: b.output_bias,
            },
            statics &statics.merger
        ) {
            vision.merger.insert(b, kernel);
        }
        Ok(())
    }
}
