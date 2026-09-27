//! Measurement of the short convolution's own classes: the segmented `u | C`
//! input projection (`short_conv_project`, a weight-streaming class) and the
//! gated taps of one row over its window (`short_conv_rows`, linear in the
//! window it reads). Its output projection is `attention_output`, measured as
//! that class.

use super::{
    bytes, failed, launch_point, multiple, size_point, Launch, MeasuredPoint, Runner, Step,
    Timed, ROW_ALIGNMENT, STEP_BANKS,
};
use magnitude_kernels::{short_conv_project, short_conv_rows};
use magnitude_state::BankComponent;
use seismic::{Element, SlabRegion};

/// Taps of the timed convolution, the current row included.
const CONVOLUTION_WIDTH: u64 = 3;

impl Runner<'_, '_, '_> {
    /// The three projection segments: the launch's rows in 3 equal parts.
    pub(super) fn short_conv_project(
        &self,
        norm: Element,
        weight: Element,
        activation: Element,
        at: Launch,
    ) -> Step<MeasuredPoint> {
        let (channels, hidden) = (multiple(at.rows / 3, ROW_ALIGNMENT), at.reduction);
        let point_bytes = bytes(weight, &[3 * channels, hidden])?;
        let device = self.device();
        let kernels = self.form::<short_conv_project::Entry>(
            &[norm, weight, activation],
            &[("M", 1), ("H", hidden), ("CH", channels), ("BS", 0), ("CS", 0), ("XS", 0)],
            move |specialization| {
                short_conv_project::native_for_device_with(
                    device,
                    short_conv_project::Elements {
                        NW: norm,
                        BW: weight,
                        CW: weight,
                        XW: weight,
                        A: activation,
                    },
                    specialization,
                )
            },
        )?;
        self.begin()?;
        let launches = self.copies(point_bytes);
        let segments = self.views(weight, hidden, channels, 3 * launches)?;
        let residual = self.zeros(Element::f32(), &[1, hidden])?;
        let norm = self.zeros(norm, &[hidden])?;
        let scale = self.zeros(Element::f32(), &[0])?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let residual = timed.bound(&residual)?;
            let norm = timed.bound(&norm)?;
            let scale = timed.bound(&scale)?;
            for launch in segments.chunks_exact(3) {
                let [b, c, x] = launch else {
                    return Err(failed("short convolution segments are not triples"));
                };
                let (b, c, x) = (timed.bound(b)?, timed.bound(c)?, timed.bound(x)?);
                let result = timed
                    .graph
                    .enqueue(
                        kernel,
                        short_conv_project::WorkflowArgs {
                            residual: residual.tensor().into(),
                            norm: norm.tensor().into(),
                            b_weight: b.tensor().into(),
                            c_weight: c.tensor().into(),
                            x_weight: x.tensor().into(),
                            eps: 1e-5,
                            b_scale: scale.tensor().into(),
                            c_scale: scale.tensor().into(),
                            x_scale: scale.tensor().into(),
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.value)?;
            }
            Ok((timed, launches))
        })?;
        Ok(launch_point(at, 3 * channels, weight, point_bytes, samples))
    }

    /// One row's gated taps over `channels` channels over the state layout's
    /// window (a one-row tape, as plain decoding plans it). Its bytes are the
    /// window's.
    pub(super) fn short_conv_rows(
        &self,
        activation: Element,
        channels: u64,
    ) -> Step<MeasuredPoint> {
        let convolution_width = CONVOLUTION_WIDTH;
        let host = |value: u64| {
            usize::try_from(value)
                .map_err(|_| failed("short convolution geometry exceeds host domain"))
        };
        let window = BankComponent::ConvWindow {
            width: host(convolution_width)?,
            channels: host(channels)?,
            dtype: seismic::DType::F32,
        }
        .spec(0)
        .map_err(failed)?;
        let window_rows = *window
            .shape
            .first()
            .ok_or_else(|| failed("short convolution window has no row axis"))?
            as u64;
        let tape_rows = window_rows - (convolution_width - 1);
        let bank_bytes = window.bytes().map_err(failed)? as u64;
        let device = self.device();
        let kernels = self.form::<short_conv_rows::Entry>(
            &[activation],
            &[
                ("M", 1),
                ("B", 1),
                ("S", STEP_BANKS),
                ("CH", channels),
                ("C", convolution_width),
                ("T", tape_rows),
            ],
            move |specialization| {
                short_conv_rows::native_for_device_with(
                    device,
                    short_conv_rows::Elements { A: activation },
                    specialization,
                )
            },
        )?;
        self.begin()?;
        let slab_banks = magnitude_state::banks_per_slab(bank_bytes).map_err(failed)? as u64;
        let regions = vec![SlabRegion {
            element: Element::f32(),
            row_shape: window.shape.iter().map(|&extent| extent as u64).collect(),
        }];
        let launches = self.slab_copies(slab_banks, STEP_BANKS, &regions)?;
        let states = self.slabbed(slab_banks, STEP_BANKS, regions, launches)?;
        let projection = self.zeros(Element::f32(), &[1, 2 * channels])?;
        let convolution = self.zeros(Element::f32(), &[channels, convolution_width])?;
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
            let segments = timed.bound(&segments)?;
            let stop = timed.bound(&stop)?;
            let previous_bank = timed.bound(&previous_bank)?;
            let previous_tape = timed.bound(&previous_tape)?;
            let following_bank = timed.bound(&following_bank)?;
            for bank in &states {
                let [window] = bank.regions.as_slice() else {
                    return Err(failed("short convolution state is not one region"));
                };
                let mut window = timed.bound(window)?;
                let result = timed
                    .graph
                    .enqueue(
                        kernel,
                        short_conv_rows::WorkflowArgs {
                            projection: projection.tensor().into(),
                            convolution: convolution.tensor().into(),
                            segments: segments.tensor().into(),
                            stop: stop.tensor().into(),
                            previous_bank: previous_bank.tensor().into(),
                            previous_tape: previous_tape.tensor().into(),
                            following_bank: following_bank.tensor().into(),
                            window: window.tensor_mut().into(),
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
}
