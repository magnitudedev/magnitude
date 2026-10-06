//! Recurrent (gated delta net) block graph: normed projection, the in-place
//! state advance publishing the gated rows, and the plain residual output
//! projection (`attention_output`). On a backend of the convolved
//! [`StepForm`], the step row classes' projection also convolves and
//! publishes the successor windows, and their step advances from the
//! convolved channels. State never moves: the
//! layer's window, delta and tape arenas are bound as ports, and per-slot
//! tables select which version (bank and tape rows) each slot reads and which
//! bank it publishes to.

use crate::programs::graph::{draft::GraphDraft, GraphError};
use crate::programs::native_target_graph::{weight, WeightPort};
use crate::{
    native::{RecurrentKernels, RecurrentStepKernels},
    ModelLoadPlan, RecurrentBinding, StateResourcePlan,
};
use magnitude_family_contracts::{WeightKind, WeightScope};
use magnitude_kernels::{
    attention_output, gated_delta_chunk, gated_delta_project, gated_delta_project_convolved,
    gated_delta_step, gated_delta_step_convolved,
};
use seismic::{BackendName, Element, NativeGraph, NativeGraphMetadata, NativePort, WorkflowTensor};

/// How a backend advances the row-sequential (step) row classes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StepForm {
    /// `gated_delta_project`, then `gated_delta_step`, which convolves and
    /// publishes the successor windows.
    Step,
    /// `gated_delta_project_convolved`, which also convolves and publishes the
    /// successor windows, then `gated_delta_step_convolved`.
    Convolved,
}

impl StepForm {
    /// The convolved form wherever the checked bundle declares both of its
    /// entries for `backend`, else the step form.
    pub(crate) fn of(backend: BackendName) -> Result<Self, String> {
        let project = seismic::generated::native_implementation_for_backend::<
            gated_delta_project_convolved::Entry,
        >(backend)
        .map_err(|error| error.to_string())?;
        let step = seismic::generated::native_implementation_for_backend::<
            gated_delta_step_convolved::Entry,
        >(backend)
        .map_err(|error| error.to_string())?;
        match (project, step) {
            (Some(_), Some(_)) => Ok(Self::Convolved),
            (None, None) => Ok(Self::Step),
            _ => Err(format!(
                "backend {backend:?} declares only one entry of the convolved recurrent step form"
            )),
        }
    }
}

pub(crate) struct RecurrentGraphEntries<'a, G: GraphDraft + 'a> {
    pub project: G::Binding<'a, gated_delta_project::Entry>,
    pub step: RecurrentStepEntries<'a, G>,
    pub chunk: G::Binding<'a, gated_delta_chunk::Entry>,
    pub output: G::Binding<'a, attention_output::Entry>,
}

/// The step row classes' entries of a [`StepForm`].
pub(crate) enum RecurrentStepEntries<'a, G: GraphDraft + 'a> {
    Step(G::Binding<'a, gated_delta_step::Entry>),
    Convolved {
        project: G::Binding<'a, gated_delta_project_convolved::Entry>,
        step: G::Binding<'a, gated_delta_step_convolved::Entry>,
    },
}

impl<'a, G: GraphDraft + 'a> Copy for RecurrentStepEntries<'a, G> {}
impl<'a, G: GraphDraft + 'a> Clone for RecurrentStepEntries<'a, G> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a> From<&'a RecurrentKernels> for RecurrentGraphEntries<'a, NativeGraph> {
    fn from(kernels: &'a RecurrentKernels) -> Self {
        Self {
            project: &kernels.project,
            step: match &kernels.step {
                RecurrentStepKernels::Step(step) => RecurrentStepEntries::Step(step),
                RecurrentStepKernels::Convolved { project, step } => {
                    RecurrentStepEntries::Convolved { project, step }
                }
            },
            chunk: &kernels.chunk,
            output: &kernels.output,
        }
    }
}

pub(crate) struct CheckedRecurrentEntries {
    project: [(&'static str, Element); 6],
    state: [(&'static str, Element); 2],
    form: StepForm,
    output: [(&'static str, Element); 2],
}

impl CheckedRecurrentEntries {
    /// The entries of `binding` on `backend` (its [`StepForm`]).
    pub(crate) fn new(binding: RecurrentBinding, backend: BackendName) -> Result<Self, String> {
        Ok(Self {
            project: [
                ("NW", binding.norm),
                ("QW", binding.qkv),
                ("GW", binding.gate),
                ("AW", binding.alpha),
                ("BW", binding.beta),
                ("A", binding.activation),
            ],
            state: [("RN", binding.recurrent_norm), ("A", binding.activation)],
            form: StepForm::of(backend)?,
            output: [("OW", binding.output), ("A", binding.activation)],
        })
    }

    pub(crate) fn entries(&self) -> RecurrentGraphEntries<'_, NativeGraphMetadata> {
        RecurrentGraphEntries {
            project: &self.project,
            step: match self.form {
                StepForm::Step => RecurrentStepEntries::Step(&self.state[..]),
                StepForm::Convolved => RecurrentStepEntries::Convolved {
                    project: &self.project[..],
                    step: &self.state[..],
                },
            },
            chunk: &self.state,
            output: &self.output,
        }
    }
}

/// Row classes at or above this size advance state with the chunked entry;
/// smaller classes use the row-sequential step. The chunked entry advances
/// slots of at most 16 rows row-sequentially too (with the step's bits), so a
/// class of at most 16 rows gains nothing from it and takes the step.
pub(crate) const CHUNKED_ROWS: u64 = 17;

/// Whether a `rows`-row class advances state with the chunked entry.
pub(crate) fn chunked(rows: u64) -> bool {
    rows >= CHUNKED_ROWS
}

/// The layer's recurrent arenas, bound to the state store's tensors per run.
#[derive(Clone)]
pub(crate) struct RecurrentStatePorts {
    pub window: NativePort,
    pub delta: NativePort,
    pub tape: NativePort,
}

/// Per-run slot tables: row segments, published row counts, the version each
/// slot reads (bank and tape rows) and the bank it publishes to.
#[derive(Clone)]
pub(crate) struct RecurrentControlPorts {
    pub segments: NativePort,
    pub stop: NativePort,
    pub previous_bank: NativePort,
    pub previous_tape: NativePort,
    pub following_bank: NativePort,
}

/// Geometry of one recurrent block at one graph class.
pub(crate) struct RecurrentBlock {
    pub rows: u64,
    pub hidden: u64,
    pub slots: u64,
    pub slab_banks: u32,
    pub key_heads: u64,
    pub value_heads: u64,
    pub width: u64,
    pub convolution_width: u64,
    pub grouped: bool,
    pub epsilon: f32,
    /// Index of this layer's window component in the store's recurrent
    /// components; its delta and tape components follow it.
    pub component_index: usize,
}

/// Ports over a recurrent layer's three bank components (window, delta or
/// state, tape) from `component_index` on, with the store's bank count and
/// the tape's rows.
pub(crate) fn bank_ports<G: GraphDraft>(
    graph: &mut G,
    state: &StateResourcePlan,
    component_index: usize,
) -> Result<(RecurrentStatePorts, u64, u64), GraphError> {
    let store = state.target_state();
    let banks = u64::try_from(
        store
            .bank_capacity
            .storage_total()
            .map_err(|error| error.to_string())?,
    )
    .map_err(|_| "recurrent bank count exceeds u64")?;
    let component = |index: usize, what: &str| {
        store
            .recurrent_components
            .get(component_index + index)
            .ok_or_else(|| format!("recurrent {what} state component is absent"))
    };
    let mut arena = |index: usize, what: &str| -> Result<NativePort, GraphError> {
        let component = component(index, what)?;
        let extents = std::iter::once(Ok(banks))
            .chain(component.shape.iter().map(|extent| {
                u64::try_from(*extent).map_err(|_| "recurrent state extent exceeds u64".to_owned())
            }))
            .collect::<Result<Vec<_>, String>>()?;
        graph.port(Element::dense(component.dtype), &extents)
    };
    let ports = RecurrentStatePorts {
        window: arena(0, "window")?,
        delta: arena(1, "delta")?,
        tape: arena(2, "tape")?,
    };
    let tape_rows = u64::try_from(
        component(2, "tape")?
            .shape
            .first()
            .copied()
            .ok_or("recurrent tape component has no row axis")?,
    )
    .map_err(|_| "recurrent tape rows exceed u64")?;
    Ok((ports, banks, tape_rows))
}

pub(crate) fn recurrent<'a, G: GraphDraft + 'a>(
    graph: &mut G,
    kernels: RecurrentGraphEntries<'a, G>,
    state: &StateResourcePlan,
    load: &ModelLoadPlan,
    scope: WeightScope,
    weights: &mut Vec<(WeightPort, NativePort)>,
    hidden: &WorkflowTensor,
    block: RecurrentBlock,
) -> Result<(WorkflowTensor, RecurrentStatePorts, RecurrentControlPorts), GraphError> {
    let input_norm = weight(graph, load, scope, WeightKind::InputNorm, weights)?;
    let qkv_weight = weight(
        graph,
        load,
        scope,
        WeightKind::RecurrentQueryKeyValue,
        weights,
    )?;
    let gate_weight = weight(graph, load, scope, WeightKind::RecurrentGate, weights)?;
    let alpha_weight = weight(graph, load, scope, WeightKind::RecurrentAlpha, weights)?;
    let beta_weight = weight(graph, load, scope, WeightKind::RecurrentBeta, weights)?;
    let convolution = weight(
        graph,
        load,
        scope,
        WeightKind::RecurrentConvolution,
        weights,
    )?;
    let rate = weight(graph, load, scope, WeightKind::RecurrentDecay, weights)?;
    let time_bias = weight(graph, load, scope, WeightKind::RecurrentTimeBias, weights)?;
    let recurrent_norm = weight(graph, load, scope, WeightKind::RecurrentNorm, weights)?;
    let output_weight = weight(graph, load, scope, WeightKind::RecurrentOutput, weights)?;

    let (
        RecurrentStatePorts {
            mut window,
            mut delta,
            mut tape,
        },
        banks,
        tape_rows,
    ) = bank_ports(graph, state, block.component_index)?;

    let projection_dimensions = [
        ("M", block.rows),
        ("H", block.hidden),
        ("NK", block.key_heads),
        ("NV", block.value_heads),
        ("W", block.width),
    ];
    let project = |graph: &mut G| {
        graph
            .enqueue(
                kernels.project,
                &projection_dimensions,
                gated_delta_project::WorkflowArgs {
                    hidden: hidden.into(),
                    input_norm: (&input_norm).into(),
                    qkv_weight: (&qkv_weight).into(),
                    gate_weight: (&gate_weight).into(),
                    alpha_weight: (&alpha_weight).into(),
                    beta_weight: (&beta_weight).into(),
                    epsilon: block.epsilon,
                },
            )
            .map(|projection| projection.value)
    };
    let dimensions = [
        ("M", block.rows),
        ("B", block.slots),
        ("S", banks),
        ("NK", block.key_heads),
        ("NV", block.value_heads),
        ("W", block.width),
        ("C", block.convolution_width),
        ("T", tape_rows),
    ];
    // Every entry that binds the slot tables gives them the chunk's geometry,
    // whichever advances this class.
    let input = |graph: &mut G, name: &str| graph.input_for(kernels.chunk, name, &dimensions);
    let controls = RecurrentControlPorts {
        segments: input(graph, "segments")?,
        stop: input(graph, "stop")?,
        previous_bank: input(graph, "previous_bank")?,
        previous_tape: input(graph, "previous_tape")?,
        following_bank: input(graph, "following_bank")?,
    };
    // The L2-norm epsilon of the q/k prologue, scaled as the model defines it.
    let norm_epsilon = block.epsilon * block.width as f32;
    let gated = if chunked(block.rows) {
        let projection = project(graph)?;
        graph
            .enqueue(
                kernels.chunk,
                &dimensions,
                gated_delta_chunk::WorkflowArgs {
                    projection: (&projection).into(),
                    convolution: (&convolution).into(),
                    rate: (&rate).into(),
                    time_bias: (&time_bias).into(),
                    recurrent_norm: (&recurrent_norm).into(),
                    segments: controls.segments.tensor().into(),
                    stop: controls.stop.tensor().into(),
                    previous_bank: controls.previous_bank.tensor().into(),
                    previous_tape: controls.previous_tape.tensor().into(),
                    following_bank: controls.following_bank.tensor().into(),
                    window: window.tensor_mut().into(),
                    delta: delta.tensor_mut().into(),
                    tape: tape.tensor_mut().into(),
                    norm_epsilon,
                    epsilon: block.epsilon,
                    grouped: block.grouped,
                    slab_banks: block.slab_banks,
                },
            )?
            .value
    } else {
        match kernels.step {
            RecurrentStepEntries::Step(step) => {
                let projection = project(graph)?;
                graph
                    .enqueue(
                        step,
                        &dimensions,
                        gated_delta_step::WorkflowArgs {
                            projection: (&projection).into(),
                            convolution: (&convolution).into(),
                            rate: (&rate).into(),
                            time_bias: (&time_bias).into(),
                            recurrent_norm: (&recurrent_norm).into(),
                            segments: controls.segments.tensor().into(),
                            stop: controls.stop.tensor().into(),
                            previous_bank: controls.previous_bank.tensor().into(),
                            previous_tape: controls.previous_tape.tensor().into(),
                            following_bank: controls.following_bank.tensor().into(),
                            window: window.tensor_mut().into(),
                            delta: delta.tensor_mut().into(),
                            tape: tape.tensor_mut().into(),
                            norm_epsilon,
                            epsilon: block.epsilon,
                            grouped: block.grouped,
                            slab_banks: block.slab_banks,
                        },
                    )?
                    .value
            }
            RecurrentStepEntries::Convolved { project, step } => {
                // The projection launch convolves and publishes the successor
                // windows; the step advances from its convolved channels.
                let projected = graph
                    .enqueue(
                        project,
                        &[
                            ("M", block.rows),
                            ("B", block.slots),
                            ("S", banks),
                            ("H", block.hidden),
                            ("NK", block.key_heads),
                            ("NV", block.value_heads),
                            ("W", block.width),
                            ("C", block.convolution_width),
                            ("T", tape_rows),
                        ],
                        gated_delta_project_convolved::WorkflowArgs {
                            hidden: hidden.into(),
                            input_norm: (&input_norm).into(),
                            qkv_weight: (&qkv_weight).into(),
                            gate_weight: (&gate_weight).into(),
                            alpha_weight: (&alpha_weight).into(),
                            beta_weight: (&beta_weight).into(),
                            convolution: (&convolution).into(),
                            segments: controls.segments.tensor().into(),
                            stop: controls.stop.tensor().into(),
                            previous_bank: controls.previous_bank.tensor().into(),
                            previous_tape: controls.previous_tape.tensor().into(),
                            following_bank: controls.following_bank.tensor().into(),
                            window: window.tensor_mut().into(),
                            epsilon: block.epsilon,
                            slab_banks: block.slab_banks,
                        },
                    )?;
                graph
                    .enqueue(
                        step,
                        &[
                            ("M", block.rows),
                            ("B", block.slots),
                            ("S", banks),
                            ("NK", block.key_heads),
                            ("NV", block.value_heads),
                            ("W", block.width),
                            ("T", tape_rows),
                        ],
                        gated_delta_step_convolved::WorkflowArgs {
                            projection: (&projected.r0).into(),
                            convolved: (&projected.r1).into(),
                            rate: (&rate).into(),
                            time_bias: (&time_bias).into(),
                            recurrent_norm: (&recurrent_norm).into(),
                            segments: controls.segments.tensor().into(),
                            stop: controls.stop.tensor().into(),
                            previous_bank: controls.previous_bank.tensor().into(),
                            previous_tape: controls.previous_tape.tensor().into(),
                            following_bank: controls.following_bank.tensor().into(),
                            delta: delta.tensor_mut().into(),
                            tape: tape.tensor_mut().into(),
                            norm_epsilon,
                            epsilon: block.epsilon,
                            grouped: block.grouped,
                            slab_banks: block.slab_banks,
                        },
                    )?
                    .value
            }
        }
    };
    let output = graph
        .enqueue(
            kernels.output,
            &[
                ("M", block.rows),
                ("D", block.hidden),
                ("Q", block.value_heads),
                ("W", block.width),
            ],
            attention_output::WorkflowArgs {
                hidden: hidden.into(),
                gated: (&gated).into(),
                output_weight: (&output_weight).into(),
            },
        )?
        .value;
    Ok((
        output,
        RecurrentStatePorts {
            window,
            delta,
            tape,
        },
        controls,
    ))
}
