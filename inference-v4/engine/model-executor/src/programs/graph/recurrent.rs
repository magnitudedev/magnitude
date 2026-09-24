//! Recurrent (gated delta net) block graph: normed projection, the in-place
//! state advance, and the gated output projection. State never moves: the
//! layer's window, delta and tape arenas are bound as ports, and per-slot
//! tables select which version (bank and tape rows) each slot reads and which
//! bank it publishes to.

use super::super::native_target_graph::weight;
use crate::{ModelLoadPlan, StateResourcePlan, native::RecurrentKernels};
use magnitude_model_contracts::{WeightKind, WeightRole, WeightScope};
use magnitude_model_kernels::{
    qwen_recurrent_chunk, qwen_recurrent_output, qwen_recurrent_project, qwen_recurrent_step,
};
use seismic::{Element, NativeGraph, NativePort, WorkflowTensor};

/// Row classes at or above this size advance state with the chunked entry;
/// smaller classes use the row-sequential step. The chunked entry advances
/// slots of at most 16 rows row-sequentially too (with the step's bits), so a
/// class of at most 16 rows gains nothing from it and takes the step.
pub(crate) const CHUNKED_ROWS: u64 = 17;

/// State components of one recurrent layer, consecutive in the store's
/// recurrent components: window, delta, tape.
pub(crate) const RECURRENT_COMPONENTS: usize = 3;

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
    pub slots: u64,
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

pub(crate) fn recurrent(
    graph: &mut NativeGraph,
    kernels: &RecurrentKernels,
    state: &StateResourcePlan,
    load: &ModelLoadPlan,
    scope: WeightScope,
    weights: &mut Vec<(WeightRole, NativePort)>,
    hidden: &WorkflowTensor,
    block: RecurrentBlock,
) -> Result<(WorkflowTensor, RecurrentStatePorts, RecurrentControlPorts), String> {
    let input_norm = weight(graph, load, scope, WeightKind::InputNorm, weights)?;
    let qkv_weight = weight(graph, load, scope, WeightKind::RecurrentQueryKeyValue, weights)?;
    let gate_weight = weight(graph, load, scope, WeightKind::RecurrentGate, weights)?;
    let alpha_weight = weight(graph, load, scope, WeightKind::RecurrentAlpha, weights)?;
    let beta_weight = weight(graph, load, scope, WeightKind::RecurrentBeta, weights)?;
    let convolution = weight(graph, load, scope, WeightKind::RecurrentConvolution, weights)?;
    let rate = weight(graph, load, scope, WeightKind::RecurrentDecay, weights)?;
    let time_bias = weight(graph, load, scope, WeightKind::RecurrentTimeBias, weights)?;
    let recurrent_norm = weight(graph, load, scope, WeightKind::RecurrentNorm, weights)?;
    let output_weight = weight(graph, load, scope, WeightKind::RecurrentOutput, weights)?;

    let store = state.target_state();
    let banks = u64::try_from(
        store
            .bank_capacity
            .storage_total()
            .map_err(|error| error.to_string())?,
    )
    .map_err(|_| "recurrent bank count exceeds u64")?;
    let mut arena = |index: usize, what: &str| -> Result<NativePort, String> {
        let component = store
            .recurrent_components
            .get(index)
            .ok_or_else(|| format!("recurrent {what} state component is absent"))?;
        let extents = std::iter::once(Ok(banks))
            .chain(component.shape.iter().map(|extent| {
                u64::try_from(*extent).map_err(|_| "recurrent state extent exceeds u64".to_owned())
            }))
            .collect::<Result<Vec<_>, String>>()?;
        graph
            .port(Element::dense(component.dtype), &extents)
            .map_err(|error| error.to_string())
    };
    let mut window = arena(block.component_index, "window")?;
    let mut delta = arena(block.component_index + 1, "delta")?;
    let mut tape = arena(block.component_index + 2, "tape")?;

    let projection = graph
        .enqueue(
            &kernels.project,
            qwen_recurrent_project::WorkflowArgs {
                hidden: hidden.into(),
                input_norm: (&input_norm).into(),
                qkv_weight: (&qkv_weight).into(),
                gate_weight: (&gate_weight).into(),
                alpha_weight: (&alpha_weight).into(),
                beta_weight: (&beta_weight).into(),
                epsilon: block.epsilon,
            },
        )
        .map_err(|error| error.to_string())?
        .value;
    let tape_rows = u64::try_from(
        store.recurrent_components[block.component_index + 2]
            .shape
            .first()
            .copied()
            .ok_or("recurrent tape component has no row axis")?,
    )
    .map_err(|_| "recurrent tape rows exceed u64")?;
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
    // The step and chunk entries share one contract, so their inputs have the
    // same geometry whichever advances this class.
    let input = |graph: &mut NativeGraph, name: &str| {
        graph
            .input_for(&kernels.step, name, &dimensions)
            .map_err(|error| error.to_string())
    };
    let controls = RecurrentControlPorts {
        segments: input(graph, "segments")?,
        stop: input(graph, "stop")?,
        previous_bank: input(graph, "previous_bank")?,
        previous_tape: input(graph, "previous_tape")?,
        following_bank: input(graph, "following_bank")?,
    };
    // The L2-norm epsilon of the q/k prologue, scaled as the model defines it.
    let norm_epsilon = block.epsilon * block.width as f32;
    let mixed = if block.rows >= CHUNKED_ROWS {
        graph
            .enqueue(
                &kernels.chunk,
                qwen_recurrent_chunk::WorkflowArgs {
                    projection: (&projection).into(),
                    convolution: (&convolution).into(),
                    rate: (&rate).into(),
                    time_bias: (&time_bias).into(),
                    segments: controls.segments.tensor().into(),
                    stop: controls.stop.tensor().into(),
                    previous_bank: controls.previous_bank.tensor().into(),
                    previous_tape: controls.previous_tape.tensor().into(),
                    following_bank: controls.following_bank.tensor().into(),
                    window: window.tensor_mut().into(),
                    delta: delta.tensor_mut().into(),
                    tape: tape.tensor_mut().into(),
                    norm_epsilon,
                    grouped: block.grouped,
                },
            )
            .map_err(|error| error.to_string())?
            .value
    } else {
        graph
            .enqueue(
                &kernels.step,
                qwen_recurrent_step::WorkflowArgs {
                    projection: (&projection).into(),
                    convolution: (&convolution).into(),
                    rate: (&rate).into(),
                    time_bias: (&time_bias).into(),
                    segments: controls.segments.tensor().into(),
                    stop: controls.stop.tensor().into(),
                    previous_bank: controls.previous_bank.tensor().into(),
                    previous_tape: controls.previous_tape.tensor().into(),
                    following_bank: controls.following_bank.tensor().into(),
                    window: window.tensor_mut().into(),
                    delta: delta.tensor_mut().into(),
                    tape: tape.tensor_mut().into(),
                    norm_epsilon,
                    grouped: block.grouped,
                },
            )
            .map_err(|error| error.to_string())?
            .value
    };
    let output = graph
        .enqueue(
            &kernels.output,
            qwen_recurrent_output::WorkflowArgs {
                hidden: hidden.into(),
                mixed: (&mixed).into(),
                projection: (&projection).into(),
                recurrent_norm: (&recurrent_norm).into(),
                output_weight: (&output_weight).into(),
                epsilon: block.epsilon,
            },
        )
        .map_err(|error| error.to_string())?
        .value;
    Ok((output, RecurrentStatePorts { window, delta, tape }, controls))
}
