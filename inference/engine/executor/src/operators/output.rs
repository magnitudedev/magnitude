//! Sublayer output tails (the contract's `OutputForm`). A residual tail is the
//! operator's own output entry, which projects and adds in one. A post-norm
//! tail projects into F32 rows (`project_rows`) and adds them through the
//! row op that normalizes each row first (`post_norm_residual`), since the
//! norm spans a whole output row that a column-tiled epilogue cannot form.

use crate::programs::graph::draft::GraphDraft;
use crate::native::{PostNormKernels, SublayerOutput};
use magnitude_kernels::{post_norm_residual, project_rows};
use seismic::{
    Element, Entry, NativeGraph, NativeGraphMetadata, NativeKernel, WorkflowTensor,
    WorkflowTensorRef,
};

/// A sublayer's output entries, with `E` the operator's own output entry.
pub(crate) enum TailEntries<'a, G: GraphDraft + 'a, E: Entry + 'a> {
    Residual(G::Binding<'a, E>),
    PostNorm(PostNormEntries<'a, G>),
}

impl<'a, G: GraphDraft + 'a, E: Entry + 'a> Copy for TailEntries<'a, G, E> {}
impl<'a, G: GraphDraft + 'a, E: Entry + 'a> Clone for TailEntries<'a, G, E> {
    fn clone(&self) -> Self {
        *self
    }
}

pub(crate) struct PostNormEntries<'a, G: GraphDraft + 'a> {
    pub project: G::Binding<'a, project_rows::Entry>,
    pub residual: G::Binding<'a, post_norm_residual::Entry>,
}

impl<'a, G: GraphDraft + 'a> Copy for PostNormEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for PostNormEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, E: Entry + 'a> From<&'a SublayerOutput<NativeKernel<E>>> for TailEntries<'a, NativeGraph, E> {
    fn from(output: &'a SublayerOutput<NativeKernel<E>>) -> Self {
        match output {
            SublayerOutput::Residual(kernel) => Self::Residual(kernel),
            SublayerOutput::PostNorm(PostNormKernels { project, residual }) => {
                Self::PostNorm(PostNormEntries { project, residual })
            }
        }
    }
}

/// Entry element assignments of a post-norm tail from the program binding:
/// the output projection's weight and activation elements, and the norm's.
pub(crate) struct CheckedPostNormEntries {
    project: [(&'static str, Element); 3],
    residual: [(&'static str, Element); 1],
}

impl CheckedPostNormEntries {
    pub(crate) fn new(weight: Element, activation: Element, norm: Element) -> Self {
        Self {
            project: [("A", activation), ("W", weight), ("Y", Element::f32())],
            residual: [("NW", norm)],
        }
    }

    pub(crate) fn entries(&self) -> PostNormEntries<'_, NativeGraphMetadata> {
        PostNormEntries {
            project: &self.project,
            residual: &self.residual,
        }
    }
}

/// One post-norm tail over `source` (`[rows, K]` activation rows) with the
/// `[N, K]` output projection `weight`: the `out_rows` rows of `residual`
/// plus the normalized projection, times `scale` (1, or the layer's output
/// scale of `OutputForm::ScaledPostNorm`), as `[out, N]` F32 rows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn post_norm<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    entries: PostNormEntries<'a, G>,
    residual: &WorkflowTensor,
    source: WorkflowTensorRef<'_>,
    weight: &WorkflowTensor,
    weight_scale: &WorkflowTensor,
    norm: &WorkflowTensor,
    out_rows: &WorkflowTensor,
    shape: PostNormShape,
    epsilon: f32,
    scale: f32,
) -> Result<WorkflowTensor, String> {
    let PostNormShape {
        rows,
        out,
        inputs,
        outputs,
    } = shape;
    let projected = graph
        .enqueue(
            entries.project,
            &[("M", out), ("K", inputs), ("N", outputs), ("WS", 0)],
            project_rows::WorkflowArgs {
                source,
                weight: weight.into(),
                weight_scale: weight_scale.into(),
            },
        )?
        .value;
    Ok(graph
        .enqueue(
            entries.residual,
            &[("M", rows), ("O", out), ("D", outputs)],
            post_norm_residual::WorkflowArgs {
                residual: residual.into(),
                projected: (&projected).into(),
                norm: norm.into(),
                out_rows: out_rows.into(),
                epsilon,
                scale,
            },
        )?
        .value)
}

/// The extents of one post-norm tail: residual rows, output rows (the rows
/// `source` holds), and the projection's input and output widths.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PostNormShape {
    pub rows: u64,
    pub out: u64,
    pub inputs: u64,
    pub outputs: u64,
}

#[cfg(test)]
mod tests {
    use crate::assessment::basis::OperationClass;
    use crate::assessment::demand::DecodeDemand;
    use crate::planning::tests::{fixture_definition, fixture_manifest};
    use crate::programs::native_target_graph::checked_target_family_storage;
    use crate::{
        resident_layout, ComponentSelection, ExecutionPath, FeedForwardProgramSlot,
        MixerProgramSlot, ModelLoadPlan, PlannedMethod, ResourceCapacity, ResourceLimits,
        ResourcePlanner, SublayerTail,
    };
    use magnitude_family_contracts::{
        ModelDefinition, OutputForm, RmsNorm, UnweightedRms, WeightDescriptor,
    };
    use magnitude_state::KvCodec;
    use seismic::BackendName;

    /// The fixture with sandwich post-norms on both sublayers, a weightless
    /// embedding RMS and a readout softcap (Muse Glimmer's forms).
    fn sandwich() -> ModelDefinition {
        let mut definition = fixture_definition();
        let hidden = definition.decoder.hidden;
        for (index, sublayer) in definition.decoder.blocks[0].sublayers.iter_mut().enumerate() {
            sublayer.output = OutputForm::PostNorm(RmsNorm {
                weight: WeightDescriptor::stored(format!("post_norm_{index}"), [hidden]),
                epsilon: 1e-8,
            });
        }
        definition.decoder.entry.norm = Some(UnweightedRms { epsilon: 1e-5 });
        definition.decoder.exit.softcap = Some(20.0);
        definition
    }

    #[test]
    fn post_norm_sublayers_plan_seal_and_assess_on_every_backend() {
        let definition = sandwich();
        crate::operators::admit(&definition, false).unwrap();
        let manifest = fixture_manifest(&definition);
        let limits = ResourceLimits {
            max_launch_rows: 64,
            max_launch_slots: 64,
            max_projected_rows: 64,
            max_images_per_request: 1,
            lookahead: false,
        };
        for backend in [
            BackendName::Cpu,
            BackendName::Metal,
            BackendName::Cuda,
            BackendName::Vulkan,
        ] {
            let load = ModelLoadPlan::derive(
                &manifest,
                &definition,
                ComponentSelection {
                    head: false,
                    vision: false,
                },
                resident_layout(ExecutionPath::Native, backend),
            )
            .unwrap();
            let plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
            let block = plan.target().blocks()[0];
            let MixerProgramSlot::Attention(attention) = block.mixer() else {
                panic!("the fixture mixes with attention");
            };
            assert!(matches!(attention.tail, SublayerTail::PostNorm { .. }));
            assert!(matches!(
                block.feed_forward(),
                Some(FeedForwardProgramSlot::Dense(dense))
                    if matches!(dense.tail, SublayerTail::PostNorm { .. })
            ));
            let state = ResourcePlanner::state_plan(
                &definition,
                &load,
                PlannedMethod::Plain,
                KvCodec::Dense,
                limits,
                ResourceCapacity {
                    domain_bytes: 1024 * 1024 * 1024,
                },
            )
            .unwrap();
            checked_target_family_storage(
                backend,
                &load,
                &definition.decoder,
                &state,
                plan.target(),
                limits,
            )
            .unwrap_or_else(|error| panic!("{backend:?}: {error}"));
            let demand = DecodeDemand::from_model(&definition, &load, KvCodec::Dense).unwrap();
            let launches = |class| {
                demand
                    .terms
                    .iter()
                    .filter(|term| term.key.class == class)
                    .map(|term| term.launches)
                    .sum::<u64>()
            };
            assert_eq!(launches(OperationClass::ProjectRows), 2);
            assert_eq!(launches(OperationClass::PostNormResidual), 2);
            assert_eq!(launches(OperationClass::AttentionOutput), 0);
            assert_eq!(launches(OperationClass::DenseOutput), 0);
        }
    }

    #[test]
    fn scaled_post_norms_plan_a_scaled_tail_with_a_layer_scale_role() {
        let mut definition = sandwich();
        let hidden = definition.decoder.hidden;
        definition.decoder.blocks[0].sublayers[1].output = OutputForm::ScaledPostNorm {
            norm: RmsNorm {
                weight: WeightDescriptor::stored("post_norm_1", [hidden]),
                epsilon: 1e-8,
            },
            layer_scale: WeightDescriptor::stored("layer_scale", [1]),
        };
        crate::operators::admit(&definition, false).unwrap();
        let manifest = fixture_manifest(&definition);
        let load = ModelLoadPlan::derive(
            &manifest,
            &definition,
            ComponentSelection {
                head: false,
                vision: false,
            },
            resident_layout(ExecutionPath::Native, BackendName::Cpu),
        )
        .unwrap();
        assert!(load.weights().any(|weight| weight.role.kind
            == magnitude_family_contracts::WeightKind::LayerScale
            && weight.shape == [1]));
        let plan = load.program_plan(&definition, KvCodec::Dense).unwrap();
        let block = plan.target().blocks()[0];
        let MixerProgramSlot::Attention(attention) = block.mixer() else {
            panic!("the fixture mixes with attention");
        };
        assert!(matches!(
            attention.tail,
            SublayerTail::PostNorm { scaled: false, .. }
        ));
        assert!(matches!(
            block.feed_forward(),
            Some(FeedForwardProgramSlot::Dense(dense))
                if matches!(dense.tail, SublayerTail::PostNorm { scaled: true, .. })
        ));
    }
}
