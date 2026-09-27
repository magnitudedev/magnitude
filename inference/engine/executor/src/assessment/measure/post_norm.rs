//! Measurement of the post-norm sublayer tail: the output projection into F32
//! rows (`project_rows`, a weight-streaming class, and the entry every weight
//! representation's per-byte time is taken through) and the row op that
//! normalizes a row into the residual (`post_norm_residual`, one launch).

use super::{
    bytes, failed, launch_point, size_point, Launch, MeasuredPoint, Runner, Step, Timed, HIDDEN,
    MAX_LAUNCHES,
};
use magnitude_kernels::{post_norm_residual, project_rows};
use seismic::Element;

impl Runner<'_, '_, '_> {
    /// One activation row projected into F32 rows: the launch's rows over
    /// its reduction.
    pub(super) fn project_rows(
        &self,
        weight: Element,
        activation: Element,
        at: Launch,
    ) -> Step<MeasuredPoint> {
        let (outputs, reduction) = (at.rows, at.reduction);
        let point_bytes = bytes(weight, &[outputs, reduction])?;
        let device = self.device();
        let kernels = self.form::<project_rows::Entry>(
            &[activation, weight],
            &[("M", 1), ("K", reduction), ("N", outputs), ("WS", 0)],
            move |specialization| {
                project_rows::native_for_device_with(
                    device,
                    project_rows::Elements {
                        A: activation,
                        W: weight,
                        Y: Element::f32(),
                    },
                    specialization,
                )
            },
        )?;
        self.begin()?;
        let launches = self.copies(point_bytes);
        let weights = self.views(weight, reduction, outputs, launches)?;
        let source = self.zeros(activation, &[1, reduction])?;
        let scale = self.zeros(Element::f32(), &[0])?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let source = timed.bound(&source)?;
            let scale = timed.bound(&scale)?;
            for weight in &weights {
                let weight = timed.bound(weight)?;
                let result = timed
                    .graph
                    .enqueue(
                        kernel,
                        project_rows::WorkflowArgs {
                            source: source.tensor().into(),
                            weight: weight.tensor().into(),
                            weight_scale: scale.tensor().into(),
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.value)?;
            }
            Ok((timed, launches))
        })?;
        Ok(launch_point(at, outputs, weight, point_bytes, samples))
    }

    /// One `HIDDEN`-wide projected row normalized into the residual: a
    /// launch-dominated class.
    pub(super) fn post_norm_residual(&self, norm: Element) -> Step<MeasuredPoint> {
        let device = self.device();
        let kernels = self.form::<post_norm_residual::Entry>(
            &[norm],
            &[("M", 1), ("O", 1), ("D", HIDDEN)],
            move |specialization| {
                post_norm_residual::native_for_device_with(
                    device,
                    post_norm_residual::Elements { NW: norm },
                    specialization,
                )
            },
        )?;
        self.begin()?;
        let launches = MAX_LAUNCHES;
        let residual = self.zeros(Element::f32(), &[1, HIDDEN])?;
        let projected = self.zeros(Element::f32(), &[1, HIDDEN])?;
        let norms = self.views(norm, HIDDEN, 1, launches)?;
        let out_rows = self.zeros(Element::i32(), &[1])?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let residual = timed.bound(&residual)?;
            let projected = timed.bound(&projected)?;
            let out_rows = timed.bound(&out_rows)?;
            for norm in &norms {
                let norm = timed.bound(&norm.reshape(&[HIDDEN]).map_err(failed)?)?;
                let result = timed
                    .graph
                    .enqueue(
                        kernel,
                        post_norm_residual::WorkflowArgs {
                            residual: residual.tensor().into(),
                            projected: projected.tensor().into(),
                            norm: norm.tensor().into(),
                            out_rows: out_rows.tensor().into(),
                            epsilon: 1e-8,
                            scale: 1.0,
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.value)?;
            }
            Ok((timed, launches))
        })?;
        Ok(size_point(bytes(norm, &[HIDDEN])?, samples))
    }
}
