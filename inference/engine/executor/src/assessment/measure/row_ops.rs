//! Measurement of the row ops of parallel branches and per-layer inputs
//! (Gemma 4): the parallel tail (`moe_tail`), the per-layer gate (a
//! weight-streaming class), the per-layer entry's row ops (`import_dense`,
//! `repack_weight`, `per_layer_inputs`, `conditioning_overlay`) and the host
//! gather and upload of a host-resident table's rows.

use super::{
    bytes, failed, launch_point, size_point, Launch, MeasuredPoint, Runner, Step, Timed, HIDDEN,
    MAX_LAUNCHES, RUNS,
};
use magnitude_kernels::{
    conditioning_overlay, import_dense, moe_tail, per_layer_gate, per_layer_inputs, repack_weight,
};
use seismic::Element;
use std::time::Instant;

/// The per-layer gate's activation code (GELU-tanh, as Gemma declares).
const GATE_ACTIVATION: i32 = 1;
/// Row bytes the table upload is timed at: a small and a large row.
const UPLOAD_BYTES: [u64; 2] = [4 << 10, 1 << 20];
/// Bytes of the host memory the uploaded rows are gathered from: beyond the
/// last-level cache, as a page-cached table is.
const UPLOAD_TABLE_BYTES: usize = 64 << 20;
/// Uploads of one sample.
const UPLOADS: usize = 64;
/// The width of each layer's per-layer input.
const PER_LAYER_WIDTH: u64 = 256;

impl Runner<'_, '_, '_> {
    /// Both branches' `HIDDEN`-wide rows normalized and their sum
    /// post-normalized into the residual: a launch-dominated class.
    pub(super) fn moe_tail(&self, norm: Element) -> Step<MeasuredPoint> {
        let device = self.device();
        let kernels = self.form::<moe_tail::Entry>(
            &[norm],
            &[("M", 1), ("D", HIDDEN)],
            move |specialization| {
                moe_tail::native_for_device_with(
                    device,
                    moe_tail::Elements { NW: norm },
                    specialization,
                )
            },
        )?;
        self.begin()?;
        let launches = MAX_LAUNCHES;
        let rows = [
            self.zeros(Element::f32(), &[1, HIDDEN])?,
            self.zeros(Element::f32(), &[1, HIDDEN])?,
            self.zeros(Element::f32(), &[1, HIDDEN])?,
        ];
        let norms = self.views(norm, HIDDEN, 3, launches)?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let [residual, dense, routed] = [
                timed.bound(&rows[0])?,
                timed.bound(&rows[1])?,
                timed.bound(&rows[2])?,
            ];
            for norms in &norms {
                let norm = |row| -> Step<_> {
                    norms
                        .slice_leading(row, row + 1)
                        .and_then(|view| view.reshape(&[HIDDEN]))
                        .map_err(failed)
                };
                let [dense_norm, routed_norm, tail_norm] =
                    [timed.bound(&norm(0)?)?, timed.bound(&norm(1)?)?, timed.bound(&norm(2)?)?];
                let result = timed
                    .graph
                    .enqueue(
                        kernel,
                        moe_tail::WorkflowArgs {
                            residual: residual.tensor().into(),
                            dense: dense.tensor().into(),
                            routed: routed.tensor().into(),
                            dense_norm: dense_norm.tensor().into(),
                            routed_norm: routed_norm.tensor().into(),
                            norm: tail_norm.tensor().into(),
                            epsilon: 1e-6,
                            scale: 1.0,
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.value)?;
            }
            Ok((timed, launches))
        })?;
        Ok(size_point(3 * bytes(norm, &[HIDDEN])?, samples))
    }

    /// The gate projection of one row into the launch's rows times one
    /// layer's per-layer input.
    pub(super) fn per_layer_gate(
        &self,
        gate: Element,
        activation: Element,
        at: Launch,
    ) -> Step<MeasuredPoint> {
        let (width, hidden) = (at.rows, at.reduction);
        let point_bytes = bytes(gate, &[width, hidden])?;
        let device = self.device();
        let kernels = self.form::<per_layer_gate::Entry>(
            &[gate, activation],
            &[("M", 1), ("D", hidden), ("L", 1), ("P", width), ("GS", 0)],
            move |specialization| {
                per_layer_gate::native_for_device_with(
                    device,
                    per_layer_gate::Elements {
                        GW: gate,
                        A: activation,
                    },
                    specialization,
                )
            },
        )?;
        self.begin()?;
        let launches = self.copies(point_bytes);
        let gates = self.views(gate, hidden, width, launches)?;
        let row = self.zeros(Element::f32(), &[1, hidden])?;
        let inputs = self.zeros(Element::f32(), &[1, 1, width])?;
        let scale = self.zeros(Element::f32(), &[0])?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let hidden = timed.bound(&row)?;
            let inputs = timed.bound(&inputs)?;
            let scale = timed.bound(&scale)?;
            for gate in &gates {
                let gate = timed.bound(gate)?;
                let result = timed
                    .graph
                    .enqueue(
                        kernel,
                        per_layer_gate::WorkflowArgs {
                            hidden: hidden.tensor().into(),
                            gate_weight: gate.tensor().into(),
                            inputs: inputs.tensor().into(),
                            layer: 0,
                            activation: GATE_ACTIVATION,
                            gate_scale: scale.tensor().into(),
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.value)?;
            }
            Ok((timed, launches))
        })?;
        Ok(launch_point(at, width, gate, point_bytes, samples))
    }

    /// One row's table rows and projected rows combined for each of
    /// `layers` layers. Its bytes are the F32 rows it writes.
    pub(super) fn per_layer_inputs(
        &self,
        table: Element,
        norm: Element,
        layers: u64,
    ) -> Step<MeasuredPoint> {
        let width = PER_LAYER_WIDTH;
        let channels = layers * width;
        let device = self.device();
        let kernels = self.form::<per_layer_inputs::Entry>(
            &[table, norm],
            &[("M", 1), ("L", layers), ("P", width)],
            move |specialization| {
                per_layer_inputs::native_for_device_with(
                    device,
                    per_layer_inputs::Elements {
                        TW: table,
                        NW: norm,
                    },
                    specialization,
                )
            },
        )?;
        self.begin()?;
        let launches = MAX_LAUNCHES;
        let gathered = self.views(table, channels, 1, launches)?;
        let projected = self.zeros(Element::f32(), &[1, channels])?;
        let norm_row = self.zeros(norm, &[width])?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            let projected = timed.bound(&projected)?;
            let norm_row = timed.bound(&norm_row)?;
            for rows in &gathered {
                let rows = timed.bound(rows)?;
                let result = timed
                    .graph
                    .enqueue(
                        kernel,
                        per_layer_inputs::WorkflowArgs {
                            gathered: rows.tensor().into(),
                            projected: projected.tensor().into(),
                            norm: norm_row.tensor().into(),
                            epsilon: 1e-6,
                            gathered_scale: 1.0,
                            projected_scale: 1.0,
                            scale: 1.0,
                        },
                    )
                    .map_err(failed)?;
                timed.export(&result.value)?;
            }
            Ok((timed, launches))
        })?;
        Ok(size_point(bytes(Element::f32(), &[1, channels])?, samples))
    }

    /// One row of `elements` values converted from `source` to
    /// `destination`: `import_dense` between dense representations, or
    /// `repack_weight` into a packed one. Its bytes are the source row's.
    pub(super) fn convert_rows(
        &self,
        packed: bool,
        source: Element,
        destination: Element,
        elements: u64,
    ) -> Step<MeasuredPoint> {
        let extents = [("B", 1), ("N", 1), ("K", elements)];
        let device = self.device();
        let launches = MAX_LAUNCHES;
        let point_bytes = bytes(source, &[1, elements])?;
        let samples = if packed {
            let kernels = self.form::<repack_weight::Entry>(
                &[source, destination],
                &extents,
                move |specialization| {
                    repack_weight::native_for_device_with(
                        device,
                        repack_weight::Elements {
                            E: source,
                            U: destination,
                        },
                        specialization,
                    )
                },
            )?;
            self.begin()?;
            let rows = self.shaped(source, &[1, 1, elements], launches)?;
            self.fastest(&kernels, |kernel| {
                let mut timed = Timed::new(device);
                for row in &rows {
                    let row = timed.bound(row)?;
                    let result = timed
                        .graph
                        .enqueue(
                            kernel,
                            repack_weight::WorkflowArgs {
                                source: row.tensor().into(),
                            },
                        )
                        .map_err(failed)?;
                    timed.export(&result.value)?;
                }
                Ok((timed, launches))
            })?
        } else {
            let kernels = self.form::<import_dense::Entry>(
                &[source, destination],
                &extents,
                move |specialization| {
                    import_dense::native_for_device_with(
                        device,
                        import_dense::Elements {
                            E: source,
                            U: destination,
                        },
                        specialization,
                    )
                },
            )?;
            self.begin()?;
            let rows = self.shaped(source, &[1, 1, elements], launches)?;
            self.fastest(&kernels, |kernel| {
                let mut timed = Timed::new(device);
                for row in &rows {
                    let row = timed.bound(row)?;
                    let result = timed
                        .graph
                        .enqueue(
                            kernel,
                            import_dense::WorkflowArgs {
                                source: row.tensor().into(),
                            },
                        )
                        .map_err(failed)?;
                    timed.export(&result.value)?;
                }
                Ok((timed, launches))
            })?
        };
        Ok(size_point(point_bytes, samples))
    }

    /// One F32 row of `elements` values copied into rows. Its bytes are the
    /// row's.
    pub(super) fn copy_rows(&self, elements: u64) -> Step<MeasuredPoint> {
        let device = self.device();
        let kernels = self.form::<conditioning_overlay::Entry>(
            &[],
            &[("M", 1), ("D", elements)],
            move |specialization| conditioning_overlay::native_for_device(device, specialization),
        )?;
        self.begin()?;
        let launches = MAX_LAUNCHES;
        let inputs = self.views(Element::f32(), elements, 1, launches)?;
        let outputs = self.views(Element::f32(), elements, 1, launches)?;
        let samples = self.fastest(&kernels, |kernel| {
            let mut timed = Timed::new(device);
            for (input, output) in inputs.iter().zip(&outputs) {
                let input = timed.bound(input)?;
                let mut output = timed.bound(output)?;
                timed
                    .graph
                    .enqueue(
                        kernel,
                        conditioning_overlay::WorkflowArgs {
                            input: input.tensor().into(),
                            out: output.tensor_mut().into(),
                        },
                    )
                    .map_err(failed)?;
            }
            Ok((timed, launches))
        })?;
        Ok(size_point(bytes(Element::f32(), &[1, elements])?, samples))
    }

    /// Host seconds of gathering one `row_bytes`-byte table row from host
    /// memory beyond the last-level cache and writing it into a graph's
    /// upload input, per upload. The input feeds the per-layer entry's
    /// conversion (`import_dense`), as the table's rows do.
    pub(super) fn table_upload(&self, row_bytes: u64) -> Step<MeasuredPoint> {
        let elements = row_bytes / 2;
        let source = Element::bf16();
        let device = self.device();
        let kernels = self.form::<import_dense::Entry>(
            &[source, Element::f32()],
            &[("B", 1), ("N", 1), ("K", elements)],
            move |specialization| {
                import_dense::native_for_device_with(
                    device,
                    import_dense::Elements {
                        E: source,
                        U: Element::f32(),
                    },
                    specialization,
                )
            },
        )?;
        let kernel = kernels
            .first()
            .ok_or_else(|| failed("import_dense formed no kernel"))?;
        self.begin()?;
        let mut graph = Timed::new(device).graph;
        let input = graph
            .input_for(kernel, "source", &[("B", 1), ("N", 1), ("K", elements)])
            .map_err(failed)?;
        let result = graph
            .enqueue(
                kernel,
                import_dense::WorkflowArgs {
                    source: input.tensor().into(),
                },
            )
            .map_err(failed)?;
        graph.export(&result.value).map_err(failed)?;
        let plan = graph.seal().map_err(failed)?;
        let mut slot = plan.new_slot().map_err(failed)?;
        let table = (0..UPLOAD_TABLE_BYTES)
            .map(|byte| byte as u8)
            .collect::<Vec<_>>();
        let row = usize::try_from(row_bytes).map_err(failed)?;
        let rows = UPLOAD_TABLE_BYTES / row;
        let mut gathered = vec![0u8; row];
        // A fixed pseudo-random row order, as a batch's tokens are.
        let mut next = 0x9e37_79b9_usize;
        let mut sample = || -> Step<f64> {
            let began = Instant::now();
            for _ in 0..UPLOADS {
                next = next.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let start = (next >> 17) % rows * row;
                gathered.copy_from_slice(&table[start..start + row]);
                slot.write_input(&input, &gathered).map_err(failed)?;
            }
            Ok(began.elapsed().as_secs_f64() / UPLOADS as f64)
        };
        // A leading sample, then the timed ones.
        sample()?;
        let samples = (0..RUNS).map(|_| sample()).collect::<Step<Vec<_>>>()?;
        Ok(size_point(row_bytes, samples))
    }

    /// The table upload at each of [`UPLOAD_BYTES`].
    pub(super) fn table_uploads(&self) -> Step<Vec<MeasuredPoint>> {
        self.each(&UPLOAD_BYTES, |row_bytes| self.table_upload(row_bytes))
    }
}
