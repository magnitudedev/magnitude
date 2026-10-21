//! LiLiCorr's corrector entries (`corrector_features`, `corrector_rows`,
//! `corrector_nodes`, `corrector_attention`, `corrector_factors`,
//! `corrector_path`) against host models of their contracts
//! (`corrector.seismic`, transcribed from llama.cpp's
//! `build_lilicorr_scores` and the greedy lattice walk of
//! `common/speculative.cpp`). The host models are checked against the
//! portable bodies at small shapes; the CPU test runs everywhere, the
//! accelerator test on every other device the host has.

use magnitude_kernels::{
    corrector_attention, corrector_factors, corrector_features, corrector_nodes, corrector_path,
    corrector_rows,
};
use seismic::{BackendName, Device, Element, NativeSpecialization, Tensor};

fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    f32::from_bits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| ((bf16_round(*value).to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn bf16_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16))
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn i32_values(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|word| i32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

/// Deterministic values in [-2, 2).
fn values(count: usize, seed: u32) -> Vec<f32> {
    (0..count)
        .map(|index| {
            ((index as u32).wrapping_mul(2654435761) ^ seed.wrapping_mul(40503)) % 2001
        } as f32
            / 500.0
            - 2.0)
        .collect()
}

fn specialization(
    statics: &[String],
    defaults: impl Fn(&NativeSpecialization) -> NativeSpecialization,
    extents: &[(&str, u64)],
) -> NativeSpecialization {
    defaults(
        &statics
            .iter()
            .fold(NativeSpecialization::new(), |fixed, name| {
                let (_, value) = extents
                    .iter()
                    .find(|(extent, _)| extent == name)
                    .unwrap_or_else(|| panic!("static {name} has no extent"));
                fixed.with_static(name.clone(), *value)
            }),
    )
}

fn tensor(device: &Device, element: Element, shape: &[usize], bytes: &[u8]) -> Tensor {
    let shape = shape
        .iter()
        .map(|extent| *extent as u64)
        .collect::<Vec<_>>();
    Tensor::from_host(device, element, &shape, bytes).unwrap()
}

fn f32_tensor(device: &Device, shape: &[usize], values: &[f32]) -> Tensor {
    tensor(device, Element::f32(), shape, &f32_bytes(values))
}

fn close(actual: &[f32], expected: &[f32], tolerance: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}");
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tolerance * (1.0 + e.abs()),
            "{what}[{index}]: {a} vs {e}"
        );
    }
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

// ---------------------------------------------------------------------------
// Host models.

/// `corrector_features`: per row and rank, the five features layer-normed
/// and projected.
#[allow(clippy::too_many_arguments)]
fn features_model(
    rows: usize,
    vocabulary: usize,
    top_k: usize,
    width: usize,
    logits: &[f32],
    unary: &[f32],
    norm_weight: &[f32],
    norm_bias: &[f32],
    up_weight: &[f32],
    up_bias: &[f32],
    epsilon: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * top_k * width];
    for row in 0..rows {
        let line = &logits[row * vocabulary..][..vocabulary];
        let maximum = line.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let total = line
            .iter()
            .map(|logit| f64::from((logit - maximum).exp()))
            .sum::<f64>();
        let normalizer = maximum + total.ln() as f32;
        for rank in 0..top_k {
            let lp = unary[row * top_k + rank] - normalizer;
            let features = [
                lp,
                lp.exp(),
                unary[row * top_k + rank] - unary[row * top_k],
                if top_k > 1 {
                    rank as f32 / (top_k - 1) as f32
                } else {
                    0.0
                },
                if rank == 0 { 1.0 } else { 0.0 },
            ];
            let mean = features.iter().sum::<f32>() / 5.0;
            let variance = features
                .iter()
                .map(|f| (f - mean) * (f - mean))
                .sum::<f32>()
                / 5.0;
            let inverse = 1.0 / (variance + epsilon).sqrt();
            let normed: Vec<f32> = (0..5)
                .map(|i| (features[i] - mean) * inverse * norm_weight[i] + norm_bias[i])
                .collect();
            for column in 0..width {
                let value = up_bias[column]
                    + (0..5)
                        .map(|i| up_weight[column * 5 + i] * normed[i])
                        .sum::<f32>();
                out[(row * top_k + rank) * width + column] = silu(value);
            }
        }
    }
    out
}

/// `corrector_rows`.
fn rows_model(
    rows: usize,
    width: usize,
    values: &[f32],
    bias: &[f32],
    residual: Option<&[f32]>,
    form: i32,
    epsilon: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * width];
    for row in 0..rows {
        let biased: Vec<f32> = (0..width)
            .map(|column| values[row * width + column] + bias[column])
            .collect();
        let length = biased
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(epsilon);
        for column in 0..width {
            let mut value = match form {
                1 => silu(biased[column]),
                2 => biased[column] / length,
                _ => biased[column],
            };
            if let Some(residual) = residual {
                value += residual[row * width + column];
            }
            out[row * width + column] = value;
        }
    }
    out
}

/// `corrector_attention`: per block, head and query node.
fn attention_model(
    blocks: usize,
    nodes: usize,
    heads: usize,
    width: usize,
    qkv: &[f32],
    input_bias: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    let joined = 3 * heads * width;
    let column = |row: usize, at: usize| qkv[row * joined + at] + input_bias[at];
    let scale = 1.0 / (width as f32).sqrt();
    let mut out = vec![0.0f32; blocks * nodes * heads * width];
    for block in 0..blocks {
        for head in 0..heads {
            for query in 0..nodes {
                let row = block * nodes + query;
                let scores: Vec<f32> = (0..nodes)
                    .map(|key| {
                        (0..width)
                            .map(|d| {
                                column(row, head * width + d)
                                    * column(block * nodes + key, heads * width + head * width + d)
                            })
                            .sum::<f32>()
                            * scale
                            + bias[(head * nodes + query) * nodes + key]
                    })
                    .collect();
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights: Vec<f32> = scores.iter().map(|s| (s - maximum).exp()).collect();
                let total = weights.iter().sum::<f32>();
                for d in 0..width {
                    let value = (0..nodes)
                        .map(|key| {
                            weights[key]
                                * column(block * nodes + key, 2 * heads * width + head * width + d)
                        })
                        .sum::<f32>();
                    out[row * heads * width + head * width + d] = value / total;
                }
            }
        }
    }
    out
}

/// `corrector_path`: each slot's greedy walk, step-major selections.
#[allow(clippy::too_many_arguments)]
fn path_model(
    slots: usize,
    rows: usize,
    top_k: usize,
    factor: usize,
    steps: usize,
    tokens: &[i32],
    incoming: &[f32],
    outgoing: &[f32],
    anchor: &[f32],
) -> Vec<i32> {
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let mut out = vec![0i32; steps * slots * 2];
    for slot in 0..slots {
        let mut previous = 0usize;
        for step in 0..steps {
            let scores: Vec<f32> = (0..top_k)
                .map(|candidate| {
                    let node = (slot * rows + step) * top_k + candidate;
                    let into = &incoming[node * factor..][..factor];
                    if step == 0 {
                        dot(into, &anchor[slot * factor..][..factor])
                    } else {
                        let from = (slot * rows + step - 1) * top_k + previous;
                        dot(into, &outgoing[from * factor..][..factor])
                    }
                })
                .collect();
            let mut sorted = scores.clone();
            sorted.sort_by(f32::total_cmp);
            assert!(
                sorted[top_k - 1] - sorted[top_k - 2] > 1.0e-3,
                "fixture has a clear winner"
            );
            let best = (0..top_k)
                .max_by(|a, b| scores[*a].total_cmp(&scores[*b]).then(b.cmp(a)))
                .unwrap();
            out[(step * slots + slot) * 2] = tokens[(slot * rows + step) * top_k + best];
            previous = best;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The portable bodies.

mod portable {
    use seismic_lang::{
        checked::{check_source, CheckedModule, SourceFile},
        entry::ElementBindings,
        failure::SourceTermination,
        interp::{Arg, Interpreter, OutcomeValue, TensorData},
        reference_math::ReferenceScalar,
        registry,
        types::DType,
    };

    pub enum Input {
        Tensor(DType, Vec<usize>, Vec<f64>),
        F32(f32),
        I32(i32),
    }

    pub fn floats(dtype: DType, shape: &[usize], values: &[f32]) -> Input {
        Input::Tensor(
            dtype,
            shape.to_vec(),
            values.iter().map(|v| f64::from(*v)).collect(),
        )
    }

    pub fn ints(shape: &[usize], values: &[i32]) -> Input {
        Input::Tensor(
            DType::I32,
            shape.to_vec(),
            values.iter().map(|v| f64::from(*v)).collect(),
        )
    }

    pub fn module() -> CheckedModule {
        let mut sources = seismic_std::sources();
        sources.push(SourceFile {
            path: "corrector.seismic".into(),
            text: include_str!("../kernels/corrector.seismic").into(),
        });
        check_source(sources).unwrap()
    }

    /// The first result of `name`, or with `mutated` the values of its input
    /// `mutated` after the run.
    pub fn run(
        module: &CheckedModule,
        name: &str,
        bindings: &[(&str, DType)],
        inputs: Vec<Input>,
        mutated: Option<usize>,
    ) -> Vec<f64> {
        let elements = bindings
            .iter()
            .fold(ElementBindings::new(), |elements, (name, dtype)| {
                elements.bind(name, registry::dense(*dtype))
            });
        let logical = module
            .entry(module.entry_named(name).unwrap(), &elements)
            .unwrap();
        let mut interpreter = Interpreter::new(&logical);
        let arguments = inputs
            .into_iter()
            .map(|input| match input {
                Input::Tensor(dtype, shape, values) => {
                    Arg::Tensor(interpreter.add_tensor(TensorData::dense(dtype, shape, values)))
                }
                Input::F32(value) => Arg::Scalar(ReferenceScalar::F32(value.to_bits())),
                Input::I32(value) => Arg::Scalar(ReferenceScalar::I32(value)),
            })
            .collect::<Vec<_>>();
        let outcome = interpreter.run(&arguments).unwrap();
        if let SourceTermination::Failed(failure) = outcome.termination() {
            panic!("{name} portable body failed: {failure}");
        }
        let tensor = match mutated {
            None => {
                let result = outcome.results().next().unwrap();
                let OutcomeValue::Tensor(tensor) = result.value() else {
                    panic!("{name} returns a tensor")
                };
                (0..tensor.element_count())
                    .map(|index| tensor.read(index).unwrap())
                    .collect()
            }
            Some(ordinal) => {
                let inputs = outcome.inputs().collect::<Vec<_>>();
                let input = inputs
                    .iter()
                    .find(|input| input.ordinal() == ordinal)
                    .unwrap();
                let tensor = input.tensor();
                (0..tensor.element_count())
                    .map(|index| tensor.read(index).unwrap())
                    .collect()
            }
        };
        tensor
    }
}

// ---------------------------------------------------------------------------
// Fixtures.

struct Features {
    rows: usize,
    vocabulary: usize,
    top_k: usize,
    width: usize,
    logits: Vec<f32>,
    unary: Vec<f32>,
    norm_weight: Vec<f32>,
    norm_bias: Vec<f32>,
    up_weight: Vec<f32>,
    up_bias: Vec<f32>,
}

impl Features {
    fn new(rows: usize, vocabulary: usize, top_k: usize, width: usize) -> Self {
        let logits = values(rows * vocabulary, 1)
            .into_iter()
            .map(|logit| logit * 4.0)
            .collect::<Vec<_>>();
        // Each row's K best logits, descending.
        let unary = (0..rows)
            .flat_map(|row| {
                let mut line = logits[row * vocabulary..][..vocabulary].to_vec();
                line.sort_by(|a, b| b.total_cmp(a));
                line.truncate(top_k);
                line
            })
            .collect();
        Self {
            rows,
            vocabulary,
            top_k,
            width,
            logits,
            unary,
            norm_weight: values(5, 2).into_iter().map(|v| 1.0 + v / 4.0).collect(),
            norm_bias: values(5, 3).into_iter().map(|v| v / 4.0).collect(),
            up_weight: values(width * 5, 4),
            up_bias: values(width, 5).into_iter().map(|v| v / 4.0).collect(),
        }
    }

    fn expected(&self) -> Vec<f32> {
        features_model(
            self.rows,
            self.vocabulary,
            self.top_k,
            self.width,
            &self.logits,
            &self.unary,
            &self.norm_weight,
            &self.norm_bias,
            &self.up_weight,
            &self.up_bias,
            1.0e-5,
        )
    }
}

/// A clear-winner lattice: each step's best candidate leads the others.
struct Path {
    slots: usize,
    rows: usize,
    top_k: usize,
    factor: usize,
    steps: usize,
    tokens: Vec<i32>,
    incoming: Vec<f32>,
    outgoing: Vec<f32>,
    anchor: Vec<f32>,
}

impl Path {
    fn new(slots: usize, rows: usize, top_k: usize, factor: usize, steps: usize) -> Self {
        let nodes = slots * rows * top_k;
        Self {
            slots,
            rows,
            top_k,
            factor,
            steps,
            tokens: (0..nodes).map(|node| 1000 + node as i32 * 7).collect(),
            incoming: values(nodes * factor, 30),
            outgoing: values(nodes * factor, 31),
            anchor: values(slots * factor, 32),
        }
    }

    fn expected(&self) -> Vec<i32> {
        path_model(
            self.slots,
            self.rows,
            self.top_k,
            self.factor,
            self.steps,
            &self.tokens,
            &self.incoming,
            &self.outgoing,
            &self.anchor,
        )
    }

    fn candidates(&self) -> Vec<i32> {
        self.tokens.iter().flat_map(|token| [*token, 0]).collect()
    }
}

#[test]
fn host_models_match_the_portable_bodies() {
    use portable::{floats, ints, module, run, Input};
    use seismic_lang::types::DType;
    let module = module();

    let case = Features::new(2, 40, 4, 6);
    let actual = run(
        &module,
        "corrector_features",
        &[("A", DType::F32)],
        vec![
            floats(DType::F32, &[2, 40], &case.logits),
            floats(DType::F32, &[2, 4], &case.unary),
            floats(DType::F32, &[5], &case.norm_weight),
            floats(DType::F32, &[5], &case.norm_bias),
            floats(DType::F32, &[6, 5], &case.up_weight),
            floats(DType::F32, &[6], &case.up_bias),
            Input::F32(1.0e-5),
        ],
        None,
    );
    let actual = actual.iter().map(|v| *v as f32).collect::<Vec<_>>();
    close(
        &actual,
        &case.expected(),
        1.0e-4,
        "portable corrector_features",
    );

    let (rows, width) = (3usize, 7usize);
    let input = values(rows * width, 6);
    let bias = values(width, 7);
    let residual = values(rows * width, 8);
    for (form, with_residual) in [(0, true), (1, false), (2, false)] {
        let residual_rows = usize::from(with_residual);
        let actual = run(
            &module,
            "corrector_rows",
            &[("Y", DType::F32)],
            vec![
                floats(DType::F32, &[rows, width], &input),
                floats(DType::F32, &[width], &bias),
                floats(
                    DType::F32,
                    &[residual_rows, rows, width],
                    if with_residual { &residual } else { &[] },
                ),
                Input::I32(form),
                Input::F32(1.0e-4),
            ],
            None,
        );
        let actual = actual.iter().map(|v| *v as f32).collect::<Vec<_>>();
        let expected = rows_model(
            rows,
            width,
            &input,
            &bias,
            with_residual.then_some(&residual[..]),
            form,
            1.0e-4,
        );
        close(
            &actual,
            &expected,
            1.0e-5,
            &format!("portable corrector_rows form {form}"),
        );
    }

    let (blocks, nodes, heads, head_width) = (2usize, 3usize, 2usize, 4usize);
    let joined = 3 * heads * head_width;
    let qkv = values(blocks * nodes * joined, 9);
    let input_bias = values(joined, 10);
    let bias = values(heads * nodes * nodes, 11);
    let actual = run(
        &module,
        "corrector_attention",
        &[("A", DType::F32)],
        vec![
            floats(DType::F32, &[blocks, nodes, 3, heads, head_width], &qkv),
            floats(DType::F32, &[3, heads, head_width], &input_bias),
            floats(DType::F32, &[heads, nodes, nodes], &bias),
        ],
        None,
    );
    let actual = actual.iter().map(|v| *v as f32).collect::<Vec<_>>();
    let expected = attention_model(blocks, nodes, heads, head_width, &qkv, &input_bias, &bias);
    close(&actual, &expected, 1.0e-5, "portable corrector_attention");

    let case = Path::new(2, 3, 4, 8, 2);
    let mut selection = vec![-1i32; case.steps * case.slots * 2];
    let actual = run(
        &module,
        "corrector_path",
        &[],
        vec![
            ints(&[2, 3, 4, 2], &case.candidates()),
            floats(DType::F32, &[2, 3, 4, 8], &case.incoming),
            floats(DType::F32, &[2, 3, 4, 8], &case.outgoing),
            floats(DType::F32, &[2, 8], &case.anchor),
            ints(&[case.steps, case.slots, 2], &selection),
        ],
        Some(4),
    );
    selection = actual.iter().map(|v| *v as i32).collect();
    assert_eq!(selection, case.expected(), "portable corrector_path");
}

// ---------------------------------------------------------------------------
// Natives.

fn features_match(device: &Device, case: &Features) {
    let implementation = corrector_features::native_implementation(device)
        .unwrap()
        .expect("corrector_features has a native implementation on every device");
    let kernel = corrector_features::native_for_device_with(
        device,
        corrector_features::Elements { A: Element::bf16() },
        &specialization(
            &implementation.statics,
            |fixed| implementation.default_specialization(fixed).unwrap(),
            &[
                ("M", case.rows as u64),
                ("V", case.vocabulary as u64),
                ("K", case.top_k as u64),
                ("H", case.width as u64),
            ],
        ),
    )
    .unwrap();
    let result = kernel
        .call(corrector_features::Args {
            logits: &f32_tensor(device, &[case.rows, case.vocabulary], &case.logits),
            unary: &f32_tensor(device, &[case.rows, case.top_k], &case.unary),
            norm_weight: &f32_tensor(device, &[5], &case.norm_weight),
            norm_bias: &f32_tensor(device, &[5], &case.norm_bias),
            up_weight: &f32_tensor(device, &[case.width, 5], &case.up_weight),
            up_bias: &f32_tensor(device, &[case.width], &case.up_bias),
            epsilon: 1.0e-5,
        })
        .unwrap()
        .value;
    let expected = case
        .expected()
        .into_iter()
        .map(bf16_round)
        .collect::<Vec<_>>();
    close(
        &bf16_values(&result.read_to_host().unwrap()),
        &expected,
        1.0e-2,
        &format!("{:?} corrector_features", device.backend()),
    );
}

fn rows_match(device: &Device) {
    let (rows, width) = (5usize, 1030usize);
    let input = values(rows * width, 12);
    let bias = values(width, 13);
    let residual = values(rows * width, 14);
    let implementation = corrector_rows::native_implementation(device)
        .unwrap()
        .expect("corrector_rows has a native implementation on every device");
    for (form, with_residual, output) in [
        (0, true, Element::f32()),
        (0, false, Element::bf16()),
        (1, false, Element::bf16()),
        (2, false, Element::f32()),
    ] {
        let residual_rows = usize::from(with_residual);
        let kernel = corrector_rows::native_for_device_with(
            device,
            corrector_rows::Elements { Y: output },
            &specialization(
                &implementation.statics,
                |fixed| implementation.default_specialization(fixed).unwrap(),
                &[
                    ("M", rows as u64),
                    ("N", width as u64),
                    ("R", residual_rows as u64),
                ],
            ),
        )
        .unwrap();
        let result = kernel
            .call(corrector_rows::Args {
                values: &f32_tensor(device, &[rows, width], &input),
                bias: &f32_tensor(device, &[width], &bias),
                residual: &f32_tensor(
                    device,
                    &[residual_rows, rows, width],
                    if with_residual { &residual } else { &[] },
                ),
                form,
                epsilon: 1.0e-4,
            })
            .unwrap()
            .value;
        let expected = rows_model(
            rows,
            width,
            &input,
            &bias,
            with_residual.then_some(&residual[..]),
            form,
            1.0e-4,
        );
        let what = format!(
            "{:?} corrector_rows form {form} {}",
            device.backend(),
            output.name()
        );
        if output == Element::f32() {
            close(
                &f32_values(&result.read_to_host().unwrap()),
                &expected,
                1.0e-5,
                &what,
            );
        } else {
            let expected = expected.into_iter().map(bf16_round).collect::<Vec<_>>();
            close(
                &bf16_values(&result.read_to_host().unwrap()),
                &expected,
                1.0e-2,
                &what,
            );
        }
    }
}

fn nodes_match(device: &Device) {
    let (slots, rows, top_k, width) = (2usize, 3usize, 4usize, 70usize);
    let nodes = slots * rows * top_k;
    let token = values(nodes * width, 15);
    let token_bias = values(width, 16);
    let pass = values(slots * rows * width, 17);
    let pass_bias = values(width, 18);
    let feature = values(nodes * width, 19);
    let feature_bias = values(width, 20);
    let slot = values(rows * width, 21);
    let rank = values(top_k * width, 22);
    let implementation = corrector_nodes::native_implementation(device)
        .unwrap()
        .expect("corrector_nodes has a native implementation on every device");
    for with_token_bias in [true, false] {
        let token_rows = usize::from(with_token_bias);
        let kernel = corrector_nodes::native_for_device(
            device,
            &specialization(
                &implementation.statics,
                |fixed| implementation.default_specialization(fixed).unwrap(),
                &[
                    ("S", slots as u64),
                    ("J", rows as u64),
                    ("K", top_k as u64),
                    ("H", width as u64),
                    ("TB", token_rows as u64),
                ],
            ),
        )
        .unwrap();
        let result = kernel
            .call(corrector_nodes::Args {
                token: &f32_tensor(device, &[slots, rows, top_k, width], &token),
                token_bias: &f32_tensor(
                    device,
                    &[token_rows, width],
                    if with_token_bias { &token_bias } else { &[] },
                ),
                pass: &f32_tensor(device, &[slots, rows, width], &pass),
                pass_bias: &f32_tensor(device, &[width], &pass_bias),
                feature: &f32_tensor(device, &[slots, rows, top_k, width], &feature),
                feature_bias: &f32_tensor(device, &[width], &feature_bias),
                slot: &f32_tensor(device, &[rows, width], &slot),
                rank: &f32_tensor(device, &[top_k, width], &rank),
            })
            .unwrap()
            .value;
        let mut expected = vec![0.0f32; nodes * width];
        for s in 0..slots {
            for j in 0..rows {
                for k in 0..top_k {
                    let node = (s * rows + j) * top_k + k;
                    for h in 0..width {
                        expected[node * width + h] = token[node * width + h]
                            + if with_token_bias { token_bias[h] } else { 0.0 }
                            + pass[(s * rows + j) * width + h]
                            + pass_bias[h]
                            + feature[node * width + h]
                            + feature_bias[h]
                            + slot[j * width + h]
                            + rank[k * width + h];
                    }
                }
            }
        }
        close(
            &f32_values(&result.read_to_host().unwrap()),
            &expected,
            1.0e-5,
            &format!("{:?} corrector_nodes TB={token_rows}", device.backend()),
        );
    }
}

fn attention_matches(device: &Device, blocks: usize, nodes: usize, heads: usize, width: usize) {
    let joined = 3 * heads * width;
    let qkv = values(blocks * nodes * joined, 23)
        .into_iter()
        .map(|value| value / 2.0)
        .collect::<Vec<_>>();
    let input_bias = values(joined, 24)
        .into_iter()
        .map(|value| value / 4.0)
        .collect::<Vec<_>>();
    let bias = values(heads * nodes * nodes, 25);
    let implementation = corrector_attention::native_implementation(device)
        .unwrap()
        .expect("corrector_attention has a native implementation on every device");
    let kernel = corrector_attention::native_for_device_with(
        device,
        corrector_attention::Elements { A: Element::bf16() },
        &specialization(
            &implementation.statics,
            |fixed| implementation.default_specialization(fixed).unwrap(),
            &[
                ("S", blocks as u64),
                ("N", nodes as u64),
                ("NH", heads as u64),
                ("W", width as u64),
            ],
        ),
    )
    .unwrap();
    let result = kernel
        .call(corrector_attention::Args {
            qkv: &f32_tensor(device, &[blocks, nodes, 3, heads, width], &qkv),
            input_bias: &f32_tensor(device, &[3, heads, width], &input_bias),
            bias: &f32_tensor(device, &[heads, nodes, nodes], &bias),
        })
        .unwrap()
        .value;
    let expected = attention_model(blocks, nodes, heads, width, &qkv, &input_bias, &bias)
        .into_iter()
        .map(bf16_round)
        .collect::<Vec<_>>();
    close(
        &bf16_values(&result.read_to_host().unwrap()),
        &expected,
        1.0e-2,
        &format!(
            "{:?} corrector_attention N={nodes} W={width}",
            device.backend()
        ),
    );
}

fn factors_match(device: &Device) {
    let (blocks, nodes, width) = (2usize, 5usize, 90usize);
    let x = values(blocks * nodes * width, 26)
        .into_iter()
        .map(bf16_round)
        .collect::<Vec<_>>();
    let anchor = values(blocks * width, 27)
        .into_iter()
        .map(bf16_round)
        .collect::<Vec<_>>();
    let implementation = corrector_factors::native_implementation(device)
        .unwrap()
        .expect("corrector_factors has a native implementation on every device");
    let kernel = corrector_factors::native_for_device_with(
        device,
        corrector_factors::Elements { A: Element::bf16() },
        &specialization(
            &implementation.statics,
            |fixed| implementation.default_specialization(fixed).unwrap(),
            &[
                ("S", blocks as u64),
                ("N", nodes as u64),
                ("H", width as u64),
            ],
        ),
    )
    .unwrap();
    let result = kernel
        .call(corrector_factors::Args {
            nodes: &tensor(
                device,
                Element::bf16(),
                &[blocks, nodes, width],
                &bf16_bytes(&x),
            ),
            anchor: &tensor(
                device,
                Element::bf16(),
                &[blocks, width],
                &bf16_bytes(&anchor),
            ),
        })
        .unwrap()
        .value;
    let mut expected = Vec::new();
    for block in 0..blocks {
        for node in 0..nodes {
            let row = &x[(block * nodes + node) * width..][..width];
            let a = &anchor[block * width..][..width];
            expected.extend(row);
            expected.extend(a);
            expected.extend(row.iter().zip(a).map(|(x, a)| bf16_round(x * a)));
        }
    }
    close(
        &bf16_values(&result.read_to_host().unwrap()),
        &expected,
        1.0e-6,
        &format!("{:?} corrector_factors", device.backend()),
    );
}

fn path_matches(device: &Device, case: &Path) {
    let implementation = corrector_path::native_implementation(device)
        .unwrap()
        .expect("corrector_path has a native implementation on every device");
    let kernel = corrector_path::native_for_device(
        device,
        &specialization(
            &implementation.statics,
            |fixed| implementation.default_specialization(fixed).unwrap(),
            &[
                ("S", case.slots as u64),
                ("J", case.rows as u64),
                ("K", case.top_k as u64),
                ("F", case.factor as u64),
                ("P", case.steps as u64),
            ],
        ),
    )
    .unwrap();
    let mut selection = tensor(
        device,
        Element::i32(),
        &[case.steps, case.slots, 2],
        &i32_bytes(&vec![-1; case.steps * case.slots * 2]),
    );
    kernel
        .call(corrector_path::Args {
            candidates: &tensor(
                device,
                Element::i32(),
                &[case.slots, case.rows, case.top_k, 2],
                &i32_bytes(&case.candidates()),
            ),
            incoming: &f32_tensor(
                device,
                &[case.slots, case.rows, case.top_k, case.factor],
                &case.incoming,
            ),
            outgoing: &f32_tensor(
                device,
                &[case.slots, case.rows, case.top_k, case.factor],
                &case.outgoing,
            ),
            anchor: &f32_tensor(device, &[case.slots, case.factor], &case.anchor),
            selection: &mut selection,
        })
        .unwrap();
    assert_eq!(
        i32_values(&selection.read_to_host().unwrap()),
        case.expected(),
        "{:?} corrector_path P={}",
        device.backend(),
        case.steps
    );
}

fn entries_match_on(device: &Device) {
    // A full-vocabulary row (Nemotron's 131072) and a small one.
    features_match(device, &Features::new(3, 1000, 8, 64));
    features_match(device, &Features::new(2, 131072, 8, 1024));
    rows_match(device);
    nodes_match(device);
    attention_matches(device, 2, 12, 3, 32);
    // The Nemotron 3.5 correctors' geometry: 7 slots of 8 candidates, 8 heads
    // of 128 columns.
    attention_matches(device, 1, 56, 8, 128);
    factors_match(device);
    for steps in [1, 3, 7] {
        path_matches(device, &Path::new(3, 7, 8, 1024, steps));
    }
}

#[test]
fn corrector_entries_match_the_host_model_on_cpu() {
    let catalog = seismic::Devices::discover(&[]).unwrap();
    entries_match_on(&catalog.open_backend(BackendName::Cpu).unwrap());
}

#[test]
fn corrector_entries_match_the_host_model_on_accelerators() {
    let catalog = seismic::Devices::discover(&[]).unwrap();
    for backend in [BackendName::Metal, BackendName::Cuda, BackendName::Vulkan] {
        let Ok(device) = catalog.open_backend(backend) else {
            continue;
        };
        entries_match_on(&device);
    }
}
