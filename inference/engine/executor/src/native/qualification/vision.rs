use super::super::*;
use super::*;
use crate::VisionKernel;
use magnitude_family_contracts::{VisionWeight, WeightKind, WeightScope};

/// Small exact values (integers of magnitude at most `span / 2`), exact in
/// every dense element.
fn pattern(count: u64, span: usize, shift: usize) -> Vec<f32> {
    (0..count as usize)
        .map(|index| ((index + shift) % span) as f32 - (span / 2) as f32)
        .collect()
}

/// The static dimension `name` of a planned kernel.
fn statics(kernel: &VisionKernel, name: &str, label: &str) -> Result<u64, CatalogFailure> {
    kernel
        .statics
        .iter()
        .find(|(bound, _)| *bound == name)
        .map(|(_, value)| *value)
        .ok_or_else(|| {
            qualification_dynamic("vision", label, format!("no static dimension {name}"))
        })
}

/// The element bound to `name`.
fn element(kernel: &VisionKernel, name: &str, label: &str) -> Result<Element, CatalogFailure> {
    kernel
        .elements
        .iter()
        .find(|(bound, _)| *bound == name)
        .map(|(_, element)| *element)
        .ok_or_else(|| qualification_dynamic("vision", label, format!("no element {name}")))
}

fn require_values(
    tensor: &Tensor,
    expected: &[f32],
    entry: &'static str,
    label: &str,
) -> Result<(), CatalogFailure> {
    if read_dense_values(tensor, entry, label)? != expected {
        return Err(qualification_dynamic(
            entry,
            label,
            "semantic fixture result mismatch",
        ));
    }
    Ok(())
}

/// `rows` copies of `row`.
fn repeated(row: &[f32], rows: u64) -> Vec<f32> {
    row.repeat(rows as usize)
}

impl<'a> QualificationView<'a> {
    /// Every vision kernel of the program once, at the projector's
    /// geometry, on fixtures whose matrices are zero, so each result is an
    /// exact sum of biases, table rows, residual inputs or uniformly
    /// attended values: the plumbing of every launch (norms, projections,
    /// clamps, rotation, attention, pooling, residuals) must hold. The
    /// arithmetic itself is covered by the kernel tests.
    pub(super) fn qualify_vision(&self, device: &Device) -> Result<(), CatalogFailure> {
        let (Some(vision_plan), Some(vision), Some(description)) = (
            self.plan.vision(),
            self.programs.vision.as_ref(),
            self.vision,
        ) else {
            return Ok(());
        };
        let rows = description.cell_rows();
        for kernel in vision_plan.kernels() {
            let label = format!("{kernel:?}");
            let label = label.as_str();
            let zeros = |entry, element, extents: &[u64]| {
                semantic_zeros(device, element, extents, entry, label)
            };
            let values = |entry, element, extents: &[u64], values: &[f32]| {
                semantic_dense_values(device, element, extents, values, entry, label)
            };
            let statics = |name| statics(kernel, name, label);
            let element = |name| element(kernel, name, label);
            let kernels = &vision.kernels;
            match kernel.entry {
                VisionEntry::PatchStem => {
                    let entry = "vision_patch_stem";
                    let (c, s, p, h, nb, nt) = (
                        statics("C")?,
                        statics("S")?,
                        statics("P")?,
                        statics("H")?,
                        statics("NB")?,
                        statics("NT")?,
                    );
                    let l = if nt == 1 {
                        let table_shape = self.weight_shape(
                            WeightScope::Vision,
                            WeightKind::Vision(VisionWeight::Position),
                            entry,
                        )?;
                        table_shape.iter().product::<u64>() / h
                    } else {
                        0
                    };
                    let bias_values = pattern(h, 7, 0);
                    let table_row = pattern(h, 5, 1);
                    let mut table_values = vec![0.0; (l * h) as usize];
                    if nt == 1 {
                        table_values[..h as usize].copy_from_slice(&table_row);
                    }
                    let result = prepared(&kernels.patch_stem, kernel, entry, label)?
                        .call(vision_patch_stem::Args {
                            pixels: &zeros(entry, Element::f32(), &[rows, c, 1 + s, p, p])?,
                            frame_weight: &zeros(entry, element("W0")?, &[h, c, p, p])?,
                            next_frame_weight: &zeros(entry, element("W1")?, &[s, h, c, p, p])?,
                            bias: &values(
                                entry,
                                element("B")?,
                                &[nb, h],
                                &repeated(&bias_values, nb),
                            )?,
                            table: &values(entry, element("PE")?, &[l, h], &table_values)?,
                            indices: &semantic_i32(
                                device,
                                &[nt, rows, 4],
                                &[0, 1, 2, 3].repeat((nt * rows) as usize),
                                entry,
                                label,
                            )?,
                            coefficients: &semantic_f32(
                                device,
                                &[nt, rows, 4],
                                &[1.0, 0.0, 0.0, 0.0].repeat((nt * rows) as usize),
                                entry,
                                label,
                            )?,
                        })
                        .map_err(|error| qualification_dynamic(entry, label, error))?
                        .value;
                    let row = table_row
                        .iter()
                        .zip(&bias_values)
                        .map(|(table, bias)| {
                            let table = if nt == 1 { *table } else { 0.0 };
                            if nb == 1 {
                                bias + table
                            } else {
                                table
                            }
                        })
                        .collect::<Vec<_>>();
                    require_values(&result, &repeated(&row, rows), entry, label)?;
                }
                VisionEntry::Norm => {
                    let entry = "vision_norm";
                    let (g, h, nw, nb, no) = (
                        statics("G")?,
                        statics("H")?,
                        statics("NW")?,
                        statics("NB")?,
                        statics("NO")?,
                    );
                    let bias_values = pattern(h, 7, 2);
                    let result = prepared(&kernels.norm, kernel, entry, label)?
                        .call(vision_norm::Args {
                            source: &semantic_f32(
                                device,
                                &[1, g, h],
                                &pattern(g * h, 9, 0),
                                entry,
                                label,
                            )?,
                            weight: &zeros(entry, element("NWE")?, &[nw, h])?,
                            bias: &values(
                                entry,
                                element("NBE")?,
                                &[nb, h],
                                &repeated(&bias_values, nb),
                            )?,
                            order: &semantic_i32(
                                device,
                                &[no, g],
                                &(0..(no * g) as i32).collect::<Vec<_>>(),
                                entry,
                                label,
                            )?,
                            epsilon: 1.0e-6,
                            centered: 1,
                            interleave: 0,
                        })
                        .map_err(|error| qualification_dynamic(entry, label, error))?
                        .value;
                    match (nw, nb) {
                        (_, 1) => {
                            require_values(&result, &repeated(&bias_values, g), entry, label)?
                        }
                        (1, _) => {
                            require_values(&result, &vec![0.0; (g * h) as usize], entry, label)?
                        }
                        _ => require_finite_nonzero(&result, entry, label)?,
                    }
                }
                VisionEntry::Linear => {
                    let entry = "vision_linear";
                    let (n, k, nb, nr, ng, nc) = (
                        statics("N")?,
                        statics("K")?,
                        statics("NB")?,
                        statics("NR")?,
                        statics("NG")?,
                        statics("NC")?,
                    );
                    let activation = element("A")?;
                    let bias_values = pattern(n, 7, 0);
                    let residual_values = pattern(rows * n, 9, 3);
                    let gate_values = pattern(rows * n, 3, 1);
                    let result = prepared(&kernels.linear, kernel, entry, label)?
                        .call(vision_linear::Args {
                            x: &values(entry, activation, &[rows, k], &pattern(rows * k, 5, 0))?,
                            weight: &zeros(entry, element("W")?, &[n, k])?,
                            bias: &values(
                                entry,
                                element("B")?,
                                &[nb, n],
                                &repeated(&bias_values, nb),
                            )?,
                            residual: &semantic_f32(
                                device,
                                &[nr, rows, n],
                                &repeated(&residual_values, nr),
                                entry,
                                label,
                            )?,
                            gate: &values(
                                entry,
                                activation,
                                &[ng, rows, n],
                                &repeated(&gate_values, ng),
                            )?,
                            minimum: &semantic_f32(
                                device,
                                &[nc],
                                &repeated(&[-1.0], nc),
                                entry,
                                label,
                            )?,
                            maximum: &semantic_f32(
                                device,
                                &[nc],
                                &repeated(&[1.0], nc),
                                entry,
                                label,
                            )?,
                            activation: 0,
                        })
                        .map_err(|error| qualification_dynamic(entry, label, error))?
                        .value;
                    let expected = (0..(rows * n) as usize)
                        .map(|index| {
                            let mut value = if nb == 1 {
                                bias_values[index % n as usize]
                            } else {
                                0.0
                            };
                            if nc == 1 {
                                value = value.clamp(-1.0, 1.0);
                            }
                            if ng == 1 {
                                value *= gate_values[index];
                            } else if nr == 1 {
                                value += residual_values[index];
                            }
                            value
                        })
                        .collect::<Vec<_>>();
                    require_values(&result, &expected, entry, label)?;
                }
                VisionEntry::Clamp => {
                    let entry = "vision_clamp";
                    let n = 64;
                    let input = pattern(rows * n, 7, 0);
                    let result = prepared(&kernels.clamp, kernel, entry, label)?
                        .call(vision_clamp::Args {
                            x: &values(entry, element("A")?, &[rows, n], &input)?,
                            minimum: &semantic_f32(device, &[1], &[-1.0], entry, label)?,
                            maximum: &semantic_f32(device, &[1], &[2.0], entry, label)?,
                        })
                        .map_err(|error| qualification_dynamic(entry, label, error))?
                        .value;
                    let expected = input
                        .iter()
                        .map(|value| value.clamp(-1.0, 2.0))
                        .collect::<Vec<_>>();
                    require_values(&result, &expected, entry, label)?;
                }
                VisionEntry::Attention => {
                    let entry = "vision_attention";
                    let (kv_heads, group, quarter, nq, nv, ws) = (
                        statics("KV")?,
                        statics("G")?,
                        statics("P")?,
                        statics("NQ")?,
                        statics("NV")?,
                        statics("WS")?,
                    );
                    let width = 4 * quarter;
                    let activation = element("A")?;
                    let shared = [rows, kv_heads, width];
                    // Every row holds the same values, so a uniform
                    // attention returns them, each query head its group's.
                    let value_row = pattern(kv_heads * width, 7, 0);
                    let attended_row = value_row
                        .chunks(width as usize)
                        .flat_map(|head| head.repeat(group as usize))
                        .collect::<Vec<_>>();
                    let result = prepared(&kernels.attention, kernel, entry, label)?
                        .call(vision_attention::Args {
                            query: &zeros(entry, activation, &[rows, kv_heads * group, width])?,
                            key: &zeros(entry, activation, &shared)?,
                            value: &values(
                                entry,
                                activation,
                                &shared,
                                &repeated(&value_row, rows),
                            )?,
                            query_norm: &semantic_ones(
                                device,
                                Element::f32(),
                                &[nq, width],
                                entry,
                                label,
                            )?,
                            key_norm: &semantic_ones(
                                device,
                                Element::f32(),
                                &[nq, width],
                                entry,
                                label,
                            )?,
                            value_norm: &semantic_ones(
                                device,
                                Element::f32(),
                                &[nv, width],
                                entry,
                                label,
                            )?,
                            coordinates: &zeros(entry, Element::i32(), &[rows, 2])?,
                            spans: &semantic_i32(
                                device,
                                &[ws, rows, 2],
                                &[0, rows as i32].repeat((ws * rows) as usize),
                                entry,
                                label,
                            )?,
                            log_base: 10000f32.ln(),
                            epsilon: 1.0e-6,
                            unit_scale: 0,
                        })
                        .map_err(|error| qualification_dynamic(entry, label, error))?
                        .value;
                    if nv == 1 {
                        require_finite_nonzero(&result, entry, label)?;
                    } else {
                        require_values(&result, &repeated(&attended_row, rows), entry, label)?;
                    }
                }
                VisionEntry::GatedConvolution => {
                    let entry = "vision_gated_convolution";
                    let (h, taps) = (statics("H")?, statics("T")?);
                    let activation = element("A")?;
                    // Each row reads itself through the centre tap and the
                    // next row through tap 0, both of weight one, and a unit
                    // gate: twice its (uniform) row.
                    let centre = taps / 2;
                    let value_row = pattern(h, 7, 0);
                    let mut weights = vec![0.0; (h * taps) as usize];
                    for channel in 0..h as usize {
                        weights[channel * taps as usize] = 1.0;
                        weights[channel * taps as usize + centre as usize] = 1.0;
                    }
                    let neighbors = (0..rows as i32)
                        .flat_map(|row| {
                            (0..taps).map(move |tap| match tap {
                                0 => (row + 1) % rows as i32,
                                tap if tap == centre => row,
                                _ => -1,
                            })
                        })
                        .collect::<Vec<_>>();
                    let result = prepared(&kernels.gated_convolution, kernel, entry, label)?
                        .call(vision_gated_convolution::Args {
                            value: &values(
                                entry,
                                activation,
                                &[rows, h],
                                &repeated(&value_row, rows),
                            )?,
                            gate: &values(
                                entry,
                                activation,
                                &[rows, h],
                                &vec![1.0; (rows * h) as usize],
                            )?,
                            weight: &values(entry, element("KE")?, &[h, taps], &weights)?,
                            neighbors: &semantic_i32(
                                device,
                                &[rows, taps],
                                &neighbors,
                                entry,
                                label,
                            )?,
                        })
                        .map_err(|error| qualification_dynamic(entry, label, error))?
                        .value;
                    let doubled = value_row
                        .iter()
                        .map(|value| 2.0 * value)
                        .collect::<Vec<_>>();
                    require_values(&result, &repeated(&doubled, rows), entry, label)?;
                }
                VisionEntry::Pool => {
                    let entry = "vision_pool";
                    let (g, h, ns) = (statics("G")?, statics("H")?, statics("NS")?);
                    let input = pattern(g * h, 5, 0);
                    let result = prepared(&kernels.pool, kernel, entry, label)?
                        .call(vision_pool::Args {
                            source: &semantic_f32(device, &[1, g, h], &input, entry, label)?,
                            standard_bias: &zeros(entry, element("SB")?, &[ns, h])?,
                            standard_scale: &semantic_ones(
                                device,
                                element("SS")?,
                                &[ns, h],
                                entry,
                                label,
                            )?,
                            weight: 1.0,
                            scale: 1.0,
                        })
                        .map_err(|error| qualification_dynamic(entry, label, error))?
                        .value;
                    let expected = (0..h as usize)
                        .map(|column| {
                            (0..g as usize)
                                .map(|member| input[member * h as usize + column])
                                .sum()
                        })
                        .collect::<Vec<f32>>();
                    require_values(&result, &expected, entry, label)?;
                }
                VisionEntry::Position => {
                    let entry = "vision_position";
                    let h = statics("H")?;
                    let table_shape = self.weight_shape(
                        WeightScope::Vision,
                        WeightKind::Vision(VisionWeight::Position),
                        entry,
                    )?;
                    let l = table_shape.iter().product::<u64>() / h;
                    let input = pattern(rows * h, 9, 0);
                    let table_row = pattern(h, 5, 1);
                    let mut table_values = vec![0.0; (l * h) as usize];
                    table_values[..h as usize].copy_from_slice(&table_row);
                    let result = prepared(&kernels.position, kernel, entry, label)?
                        .call(vision_position::Args {
                            source: &semantic_f32(device, &[rows, h], &input, entry, label)?,
                            table: &values(entry, element("PE")?, &[l, h], &table_values)?,
                            indices: &semantic_i32(
                                device,
                                &[rows, 4],
                                &[0, 1, 2, 3].repeat(rows as usize),
                                entry,
                                label,
                            )?,
                            coefficients: &semantic_f32(
                                device,
                                &[rows, 4],
                                &[1.0, 0.0, 0.0, 0.0].repeat(rows as usize),
                                entry,
                                label,
                            )?,
                        })
                        .map_err(|error| qualification_dynamic(entry, label, error))?
                        .value;
                    let expected = input
                        .iter()
                        .enumerate()
                        .map(|(index, value)| value + table_row[index % h as usize])
                        .collect::<Vec<_>>();
                    require_values(&result, &expected, entry, label)?;
                }
                VisionEntry::PostNormResidual => {
                    let entry = "post_norm_residual";
                    let width = description.hidden;
                    let input = pattern(rows * width, 9, 0);
                    let result = prepared(&kernels.post_norm, kernel, entry, label)?
                        .call(post_norm_residual::Args {
                            residual: &semantic_f32(device, &[rows, width], &input, entry, label)?,
                            projected: &zeros(entry, Element::f32(), &[rows, width])?,
                            norm: &semantic_ones(device, element("NW")?, &[width], entry, label)?,
                            out_rows: &semantic_i32(
                                device,
                                &[rows],
                                &(0..rows as i32).collect::<Vec<_>>(),
                                entry,
                                label,
                            )?,
                            epsilon: 1.0e-6,
                            scale: 1.0,
                        })
                        .map_err(|error| qualification_dynamic(entry, label, error))?
                        .value;
                    require_values(&result, &input, entry, label)?;
                }
            }
        }
        Ok(())
    }
}

fn prepared<'a, E: seismic::Entry>(
    kernels: &'a std::collections::HashMap<VisionKernel, NativeKernel<E>>,
    kernel: &VisionKernel,
    entry: &'static str,
    label: &str,
) -> Result<&'a NativeKernel<E>, CatalogFailure> {
    kernels
        .get(kernel)
        .ok_or_else(|| qualification_dynamic(entry, label, "the kernel was not prepared"))
}
