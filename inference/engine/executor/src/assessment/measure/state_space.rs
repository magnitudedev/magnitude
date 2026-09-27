//! Measurement of the state-space (Mamba-2) mixer's own classes: the state
//! advance of one row over its bank (`state_space_step`) and the gated group
//! norm of one row (`state_space_gate`), both linear in the bytes they touch.
//! Its projection row and output projection are `attention_project` and
//! `attention_output`, measured as those classes.

use super::{failed, size_point, MeasuredPoint, Runner, Step, Timed, STEP_BANKS};
use magnitude_kernels::{state_space_gate, state_space_step};
use magnitude_state::BankComponent;
use seismic::{Element, SlabRegion};

/// The timed mixer's geometry beside its heads: channels per head, groups
/// sharing `B` and `C`, state columns and convolution taps.
const HEAD_WIDTH: u64 = 64;
const GROUPS: u64 = 8;
const STATE: u64 = 128;
const CONVOLUTION_WIDTH: u64 = 4;

impl Runner<'_, '_, '_> {
    /// One row's state advance of `heads` heads, over the state layout's
    /// bank components (a one-row tape, as plain decoding plans it). Its
    /// bytes are the bank's.
    pub(super) fn state_space_step(
        &self,
        activation: Element,
        heads: u64,
    ) -> Step<MeasuredPoint> {
        let (head_width, groups, state, convolution_width) =
            (HEAD_WIDTH, GROUPS, STATE, CONVOLUTION_WIDTH);
        let host = |value: u64| {
            usize::try_from(value).map_err(|_| failed("state-space geometry exceeds host domain"))
        };
        let inner = heads * head_width;
        let channels = inner + 2 * groups * state;
        // The bank as the state layout forms it: the window keeps raw
        // projection rows in activation precision.
        let [window, ssm, tape] = [
            BankComponent::ConvWindow {
                width: host(convolution_width)?,
                channels: host(channels)?,
                dtype: activation
                    .dtype()
                    .ok_or_else(|| failed("a decoder activation is dense"))?,
            },
            BankComponent::SsmState {
                heads: host(heads)?,
                head_width: host(head_width)?,
                state_width: host(state)?,
            },
            BankComponent::SsmTape {
                heads: host(heads)?,
                head_width: host(head_width)?,
                groups: host(groups)?,
                state_width: host(state)?,
            },
        ]
        .map(|component| component.spec(0).map_err(failed));
        let (window, ssm, tape) = (&window?, &ssm?, &tape?);
        let tape_rows = *tape
            .shape
            .first()
            .ok_or_else(|| failed("state-space tape has no row axis"))? as u64;
        let bank_bytes = [window, ssm, tape]
            .iter()
            .try_fold(0u64, |total, component| {
                total
                    .checked_add(component.bytes().map_err(failed)? as u64)
                    .ok_or_else(|| failed("state-space bank bytes overflow"))
            })?;
        let device = self.device();
        let kernels = self.form::<state_space_step::Entry>(
            &[activation],
            &[
                ("M", 1),
                ("B", 1),
                ("S", STEP_BANKS),
                ("NH", heads),
                ("P", head_width),
                ("G", groups),
                ("N", state),
                ("C", convolution_width),
                ("T", tape_rows),
            ],
            move |specialization| {
                state_space_step::native_for_device_with(
                    device,
                    state_space_step::Elements { A: activation },
                    specialization,
                )
            },
        )?;
        self.begin()?;
        let slab_banks = magnitude_state::banks_per_slab(bank_bytes).map_err(failed)? as u64;
        let regions = [window, ssm, tape]
            .iter()
            .map(|component| SlabRegion {
                element: Element::dense(component.dtype),
                row_shape: component
                    .shape
                    .iter()
                    .map(|&extent| extent as u64)
                    .collect(),
            })
            .collect::<Vec<_>>();
        let launches = self.slab_copies(slab_banks, STEP_BANKS, &regions)?;
        let states = self.slabbed(slab_banks, STEP_BANKS, regions, launches)?;
        let projection = self.zeros(activation, &[1, inner + channels + heads])?;
        let convolution = self.zeros(Element::f32(), &[channels, convolution_width])?;
        let convolution_bias = self.zeros(Element::f32(), &[channels])?;
        let rate = self.zeros(Element::f32(), &[heads])?;
        let time_bias = self.zeros(Element::f32(), &[heads])?;
        let skip = self.zeros(Element::f32(), &[heads])?;
        // One slot of one row reads the pristine bank 0 and publishes bank 1
        // after its row; the terminal segment row closes the table.
        let segments = self.i32s(&[2, 2], &[0, 1, 1, 1])?;
        let stop = self.i32s(&[1], &[1])?;
        let previous_bank = self.i32s(&[1], &[0])?;
        let previous_tape = self.i32s(&[1], &[0])?;
        let following_bank = self.i32s(&[1], &[1])?;
        let slab_banks =
            u32::try_from(slab_banks).map_err(|_| failed("bank slab count exceeds u32"))?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let projection = timed.bound(&projection)?;
            let convolution = timed.bound(&convolution)?;
            let convolution_bias = timed.bound(&convolution_bias)?;
            let rate = timed.bound(&rate)?;
            let time_bias = timed.bound(&time_bias)?;
            let skip = timed.bound(&skip)?;
            let segments = timed.bound(&segments)?;
            let stop = timed.bound(&stop)?;
            let previous_bank = timed.bound(&previous_bank)?;
            let previous_tape = timed.bound(&previous_tape)?;
            let following_bank = timed.bound(&following_bank)?;
            for bank in &states {
                let [window, ssm, tape] = bank.regions.as_slice() else {
                    return Err(failed("state-space state is not three regions"));
                };
                let mut window = timed.bound(window)?;
                let mut ssm = timed.bound(ssm)?;
                let mut tape = timed.bound(tape)?;
                let result = timed
                    .graph
                    .enqueue(
                        kernel,
                        state_space_step::WorkflowArgs {
                            projection: projection.tensor().into(),
                            convolution: convolution.tensor().into(),
                            convolution_bias: convolution_bias.tensor().into(),
                            rate: rate.tensor().into(),
                            time_bias: time_bias.tensor().into(),
                            skip: skip.tensor().into(),
                            segments: segments.tensor().into(),
                            stop: stop.tensor().into(),
                            previous_bank: previous_bank.tensor().into(),
                            previous_tape: previous_tape.tensor().into(),
                            following_bank: following_bank.tensor().into(),
                            window: window.tensor_mut().into(),
                            state: ssm.tensor_mut().into(),
                            tape: tape.tensor_mut().into(),
                            slab_banks,
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.value)?;
            }
            Ok((timed, launches))
        })?;
        Ok(size_point(bank_bytes, samples))
    }

    /// One row's gated group norm of a mixer of `heads` heads, normed in
    /// its 8 groups. Its bytes are the F32 norm weight's.
    pub(super) fn state_space_gate(
        &self,
        activation: Element,
        heads: u64,
    ) -> Step<MeasuredPoint> {
        let (groups, width) = (GROUPS, HEAD_WIDTH);
        let heads_per_group = (heads / groups).max(1);
        let projection_width = 2 * heads * width + 2 * groups * STATE + heads;
        let heads = heads_per_group;
        let device = self.device();
        let kernels = self.form::<state_space_gate::Entry>(
            &[activation],
            &[
                ("M", 1),
                ("G", groups),
                ("U", heads),
                ("P", width),
                ("J", projection_width),
            ],
            move |specialization| {
                state_space_gate::native_for_device_with(
                    device,
                    state_space_gate::Elements { A: activation },
                    specialization,
                )
            },
        )?;
        self.begin()?;
        let launches = super::MAX_LAUNCHES;
        let mixed = self.zeros(activation, &[1, groups * heads, width])?;
        let projection = self.zeros(activation, &[1, projection_width])?;
        let norm = self.zeros(Element::f32(), &[groups, heads, width])?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let mixed = timed.bound(&mixed)?;
            let projection = timed.bound(&projection)?;
            let norm = timed.bound(&norm)?;
            for _ in 0..launches {
                let result = timed
                    .graph
                    .enqueue(
                        kernel,
                        state_space_gate::WorkflowArgs {
                            mixed: mixed.tensor().into(),
                            projection: projection.tensor().into(),
                            state_norm: norm.tensor().into(),
                            epsilon: 1e-5,
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.value)?;
            }
            Ok((timed, launches))
        })?;
        Ok(size_point(4 * groups * heads * width, samples))
    }
}
