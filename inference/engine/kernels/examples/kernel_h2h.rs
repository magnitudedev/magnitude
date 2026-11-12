//! The CPU kernel head-to-head's Magnitude side: `project_rows` (a stage
//! launch, then the rows launch) through the CPU backend's pool, timed per
//! quant × shape × activation rows, at every `ROWS` × `INT8` configuration
//! the declaration offers, at one pool size a process. llama.cpp's side is
//! `campaigns/cpu-kernels/scripts/ggml-build.sh`'s `ggml-mul-mat`, which
//! reads the same data files and prints the same JSONL.
//!
//! `kernel_h2h --write-data DIR [--quants a,b] [--shapes NxK,..]` writes the
//! shared data: GGUF blocks of each quant × shape (`w-QUANT-NxK.bin`:
//! random codes, scale fields set so a row dot with unit-normal activations
//! is of order one) and 512 unit-normal F32
//! activation rows per K (`x-K.bin`). `kernel_h2h --data DIR --threads T
//! --rotation-bytes B --host H --cpu C --placement P [--quants a,b]
//! [--shapes NxK,..] [--rows 1,..] [--search full|forms|default]` prints one
//! line per cell and choice.
//!
//! Cold weights: a call's weights are one of `ceil(B / weight bytes)`
//! copies, called in turn in one pass (one pool job, a barrier between
//! calls), so with B at least four times the host's last-level cache no
//! call finds its weights cached, as in a model step. The 512-row prefill
//! reads its weights once a call and is arithmetic-bound, so it takes one
//! copy. A sample is one pass; its time is per call.
//!
//! Search: one pass of each form (`INT8`) at the default `ROWS`; the forms
//! within twice the fastest are searched over every `ROWS`, one pass each,
//! and those within 1.25 times the fastest get two more; the fastest by its
//! median, the default, and the default's ROWS in the other form are then
//! timed over 20 passes; `--search forms` times the default and that other
//! form alone, `--search default` the default alone. `--seconds S` times each
//! printed configuration for about `S` seconds more first (`seismic::measure`,
//! a `Stop::Count` of 50 ms samples), so a sampling profiler's stops fall in
//! the timed cell, not in the seconds of setup before it: a line on stderr,
//! `repeating ..`, marks its start. Each printed
//! configuration is checked against an f64 reference on 64 of its weight
//! rows (`rel_err`, the relative RMS error), and must hold every value
//! within the int8-activation bound (`within`): half a 256-value block's
//! quantization step times |w| per product, plus 1e-4 of the magnitude sum.

use magnitude_ggml::Encoding;
use magnitude_kernels::project_rows;
use seismic::measure::{measure, Measure, Pending, Stop, Work};
use seismic::{BackendName, Device, Element, GgufScales, NativeSpecialization, OperandRecipe, Tensor};

/// Each quant: its name, encoding, and scale fields (byte offset, the code
/// magnitude the field's f16 divides a block's reach by).
const QUANTS: [(&str, Encoding, &[(usize, f32)]); 5] = [
    ("q4_k", Encoding::Q4K, &[(0, 945.0), (2, 63.0)]),
    ("q5_k", Encoding::Q5K, &[(0, 1953.0), (2, 63.0)]),
    ("q6_k", Encoding::Q6K, &[(208, 4096.0)]),
    ("q8_0", Encoding::Q8_0, &[(0, 127.0)]),
    ("iq4_nl", Encoding::Iq4Nl, &[(0, 128.0)]),
];
/// Weight rows × K: Qwen3.6-35B-A3B's expert gate_up and down, a dense 4B
/// projection, and the standard 14336 × 4096.
const SHAPES: [(usize, usize); 4] = [(1024, 2048), (2048, 512), (9728, 2560), (14336, 4096)];
const ROWS: [usize; 9] = [1, 2, 3, 4, 5, 6, 7, 8, PREFILL];
const PREFILL: usize = 512;
const SAMPLES: usize = 20;
/// Device seconds one sample of a `--seconds` repeat covers at least.
const SAMPLE_SECONDS: f64 = 0.05;
/// Weight rows the correctness check reads: eight groups of eight.
const CHECKED: usize = 64;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let z = (self.0 ^ (self.0 >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        let z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn source(encoding: Encoding) -> Element {
    Element::gguf_packet_source(encoding, GgufScales::Own).expect("a GGUF source")
}

/// Bytes and values of one GGUF block of `encoding`.
fn block(encoding: Encoding) -> (usize, usize) {
    let values = source(encoding).logical_group().expect("a source has packets");
    (source(encoding).canonical_byte_len(&[1, values]).expect("a packet's bytes") as usize, values as usize)
}

fn write_data(dir: &std::path::Path, quants: &[String], shapes: &[(usize, usize)]) {
    std::fs::create_dir_all(dir).expect("data directory");
    for (index, (name, encoding, fields)) in QUANTS.iter().enumerate().filter(|(_, q)| quants.iter().any(|s| s == q.0)) {
        let (bytes, values) = block(*encoding);
        for &(n, k) in shapes {
            let mut rng = Rng((index * 7919 + n * 31 + k) as u64);
            let scale = 2.0 / (k as f32).sqrt();
            let mut data = vec![0u8; n * k / values * bytes];
            for block in data.chunks_exact_mut(bytes) {
                block.iter_mut().for_each(|byte| *byte = rng.next() as u8);
                let reach = scale * (0.5 + rng.unit() as f32);
                for &(at, magnitude) in *fields {
                    block[at..at + 2].copy_from_slice(&seismic::f16_bits(reach / magnitude).to_le_bytes());
                }
            }
            std::fs::write(dir.join(format!("w-{name}-{n}x{k}.bin")), data).expect("weights file");
        }
    }
    let mut ks = shapes.iter().map(|shape| shape.1).collect::<Vec<_>>();
    ks.sort_unstable();
    ks.dedup();
    for k in ks {
        let mut rng = Rng(k as u64);
        let normal = (0..PREFILL * k).map(|_| {
            let (u, v) = (rng.unit().max(f64::MIN_POSITIVE), rng.unit());
            ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32
        });
        std::fs::write(dir.join(format!("x-{k}.bin")), normal.flat_map(f32::to_le_bytes).collect::<Vec<_>>())
            .expect("activations file");
    }
}

/// The f64 reference and the int8-activation bound of the checked rows,
/// `[activation row][checked row]`, and the checked rows' indices.
struct Check {
    rows: Vec<usize>,
    reference: Vec<f64>,
    bound: Vec<f64>,
}

impl Check {
    fn new(element: Element, encoding: Encoding, gguf: &[u8], x: &[f32], n: usize, k: usize) -> Self {
        let row_bytes = gguf.len() / n;
        // Eight groups of eight rows spread over the matrix.
        let groups = (0..CHECKED / 8).map(|g| g * n / 8 / 8 * 8);
        let mut rows = Vec::new();
        let mut weights = Vec::new();
        for first in groups {
            let packed = element
                .repack_host(source(encoding), &[8, k as u64], &gguf[first * row_bytes..(first + 8) * row_bytes])
                .expect("checked rows repack");
            weights.extend(element.decode_host(&[8, k as u64], &packed).expect("checked rows decode"));
            rows.extend(first..first + 8);
        }
        let (mut reference, mut bound) = (Vec::new(), Vec::new());
        for x in x.chunks_exact(k) {
            let steps = x.chunks(256).map(|run| run.iter().fold(0f32, |m, v| m.max(v.abs())) as f64 / 127.0).collect::<Vec<_>>();
            for w in weights.chunks_exact(k) {
                let exact = w.iter().zip(x).map(|(w, x)| w * *x as f64).sum::<f64>();
                let rounding = w.iter().enumerate().map(|(i, w)| w.abs() * steps[i / 256] * 0.5).sum::<f64>();
                let magnitude = w.iter().zip(x).map(|(w, x)| (w * *x as f64).abs()).sum::<f64>();
                reference.push(exact);
                bound.push(rounding + 1e-4 * magnitude);
            }
        }
        Self { rows, reference, bound }
    }
    /// Relative RMS error of `y` (activation-major, `n` columns) and whether
    /// every checked value is within its bound.
    fn judge(&self, y: &[f32], n: usize) -> (f64, bool) {
        let (mut error, mut scale, mut within) = (0.0, 0.0, true);
        for (a, y) in y.chunks_exact(n).enumerate() {
            for (c, row) in self.rows.iter().enumerate() {
                let (exact, bound) = (self.reference[a * CHECKED + c], self.bound[a * CHECKED + c]);
                let difference = y[*row] as f64 - exact;
                (error, scale) = (error + difference * difference, scale + exact * exact);
                within &= difference.abs() <= bound;
            }
        }
        ((error / scale).sqrt(), within)
    }
}

fn arguments() -> std::collections::HashMap<String, String> {
    let words = std::env::args().skip(1).collect::<Vec<_>>();
    words.chunks(2).map(|pair| (pair[0].trim_start_matches("--").to_owned(), pair.get(1).cloned().unwrap_or_default())).collect()
}

fn f32_tensor(device: &Device, shape: &[u64], values: &[f32]) -> Tensor {
    Tensor::from_host(device, Element::f32(), shape, &values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
        .expect("an F32 tensor")
}

fn quantiles(mut ms: Vec<f64>) -> [f64; 3] {
    ms.sort_by(f64::total_cmp);
    [0.5, 0.1, 0.9].map(|q| ms[((ms.len() - 1) as f64 * q).round() as usize])
}

fn main() {
    let options = arguments();
    let selected = |name: &str, all: Vec<String>| match options.get(name) {
        Some(list) => list.split(',').map(str::to_owned).collect::<Vec<_>>(),
        None => all,
    };
    let shapes = selected("shapes", SHAPES.iter().map(|(n, k)| format!("{n}x{k}")).collect());
    let quants = selected("quants", QUANTS.iter().map(|q| q.0.to_owned()).collect());
    if let Some(dir) = options.get("write-data") {
        let shapes = shapes
            .iter()
            .map(|shape| {
                let (n, k) = shape.split_once('x').expect("a shape is NxK");
                (n.parse().expect("N is a count"), k.parse().expect("K is a count"))
            })
            .collect::<Vec<_>>();
        return write_data(std::path::Path::new(dir), &quants, &shapes);
    }
    let option = |name: &str| options.get(name).unwrap_or_else(|| panic!("--{name} is required")).clone();
    let threads = option("threads").parse::<std::num::NonZeroUsize>().expect("--threads is a positive count");
    // The pool every CPU device of the process shares (`seismic_cpu::open_host`),
    // held so the device opened next runs on exactly `threads` participants.
    let pool = seismic_cpu::open_host(threads).expect("the host pool");
    assert_eq!(pool.executor.workers(), threads.get(), "the pool has the asked participants");
    let device = seismic::Devices::discover(&[]).unwrap().open_backend(BackendName::Cpu).expect("the CPU device");
    let rotation: usize = option("rotation-bytes").parse().expect("--rotation-bytes is a byte count");
    let data = std::path::PathBuf::from(option("data"));
    // `--search`: `full` (the default) searches every ROWS x INT8; `forms`
    // times the default and its ROWS in the other form; `default` the
    // default alone. Only the configurations timed are formed.
    let search = options.get("search").map_or("full", String::as_str).to_owned();
    assert!(matches!(search.as_str(), "full" | "forms" | "default"), "--search is full, forms or default");
    let seconds = options.get("seconds").map(|s| s.parse::<f64>().expect("--seconds is a duration"));
    let rows: Vec<usize> = selected("rows", ROWS.iter().map(usize::to_string).collect()).iter().map(|r| r.parse().unwrap()).collect();
    let prefix = format!(
        "\"host\":\"{}\",\"cpu\":\"{}\",\"placement\":\"{}\",\"engine\":\"magnitude\",\"threads\":{threads}",
        option("host"), option("cpu"), option("placement")
    );
    let empty = f32_tensor(&device, &[0], &[]);
    for &(name, encoding, _) in QUANTS.iter().filter(|q| quants.iter().any(|s| s == q.0)) {
        for shape in &shapes {
            let (n, k) = shape.split_once('x').map(|(n, k)| (n.parse::<usize>().unwrap(), k.parse::<usize>().unwrap())).expect("a shape NxK");
            let gguf = std::fs::read(data.join(format!("w-{name}-{shape}.bin"))).expect("weights file");
            let x = std::fs::read(data.join(format!("x-{k}.bin"))).expect("activations file");
            let x = x.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
            let element = source(encoding).converted(device.backend().resident_layout(), OperandRecipe::Rounded).expect("a CPU layout");
            let packed = element.repack_host(source(encoding), &[n as u64, k as u64], &gguf).expect("repack");
            let copies = rotation.div_ceil(gguf.len()).max(1);
            let weights = (0..copies).map(|_| Tensor::from_host(&device, element, &[n as u64, k as u64], &packed).unwrap()).collect::<Vec<_>>();
            let check = Check::new(element, encoding, &gguf, &x, n, k);
            let elements = || project_rows::Elements { A: Element::f32(), W: element, Y: Element::f32() };
            let offered = seismic::generated::native_offered::<project_rows::Entry>(&device, project_rows::native_entry_with(elements()).elements())
                .unwrap()
                .expect("a CPU project_rows");
            let statics = [("G", 1), ("K", k), ("N", n), ("WS", 0), ("WB", 0)]
                .into_iter()
                .filter(|(name, _)| offered.implementation().statics.iter().any(|declared| declared == name))
                .fold(NativeSpecialization::new(), |spec, (name, value)| spec.with_static(name, value as u64));
            let default = offered.default_specialization(&statics).expect("a default configuration");
            let chosen = |rows: u64, int8: u64| default.clone().with_param("ROWS", rows).with_param("INT8", int8);
            let (rows_default, int8_default) = (default.param("ROWS").unwrap(), default.param("INT8").unwrap());
            let timed = |(r, i): (u64, u64)| match search.as_str() {
                "full" => true,
                "forms" => r == rows_default,
                _ => (r, i) == (rows_default, int8_default),
            };
            let configurations = [8, 4, 2, 1].into_iter().flat_map(|r| [(r, 1), (r, 0)]).filter(|c| timed(*c)).filter_map(|(r, i)| {
                project_rows::native_for_device_with(&device, elements(), &chosen(r, i)).ok().map(|kernel| ((r, i), kernel))
            });
            let configurations = configurations.collect::<std::collections::BTreeMap<_, _>>();
            for &m in &rows {
                let activations = f32_tensor(&device, &[m as u64, k as u64], &x[..m * k]);
                let calls = if m <= 8 { copies } else { 1 };
                let args = |weight| project_rows::Args { source: &activations, weight, weight_scale: &empty, weight_bound: &empty };
                let points = configurations
                    .iter()
                    .map(|(config, kernel)| (*config, kernel.point(weights[..calls].iter().map(args).collect()).expect("a placed rotation")))
                    .collect::<std::collections::BTreeMap<_, _>>();
                let pass = |config| {
                    let pending = Work::submit(&points[&config], 1).expect("a pass");
                    pending.seconds().expect("a timed pass") * 1e3 / calls as f64
                };
                // Config `config` timed to SAMPLES passes in all, checked, and
                // printed as `choice`.
                let report = |choice: &str, config: (u64, u64), taken: &mut Vec<f64>| {
                    if let Some(seconds) = seconds {
                        // The line a profiler waits for before it samples.
                        eprintln!("repeating {name} {shape} rows {m} for {seconds} s");
                        // Samples of at least `SAMPLE_SECONDS` each, as many as fill `seconds`.
                        let samples = std::num::NonZeroUsize::new((seconds / SAMPLE_SECONDS).ceil().max(1.0) as usize).unwrap();
                        let how = Measure { stop: Stop::Count(samples), min_sample_seconds: SAMPLE_SECONDS };
                        measure(&points[&config], &how, None).expect("a timed repeat");
                    }
                    while taken.len() < SAMPLES {
                        taken.push(pass(config));
                    }
                    let [median, p10, p90] = quantiles(taken.clone());
                    let y = configurations[&config].call(args(&weights[0])).expect("a checked call").value.read_to_host().unwrap();
                    let y = y.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
                    let (rel_err, within) = check.judge(&y, n);
                    println!(
                        "{{{prefix},\"quant\":\"{name}\",\"shape\":\"{shape}\",\"rows\":{m},\"config\":\"ROWS={},INT8={}\",\"choice\":\"{choice}\",\"copies\":{calls},\"ms_median\":{median:.5},\"ms_p10\":{p10:.5},\"ms_p90\":{p90:.5},\"gbps\":{:.3},\"rel_err\":{rel_err:.3e},\"within\":{within}}}",
                        config.0, config.1, gguf.len() as f64 / median / 1e6
                    );
                };
                pass((rows_default, int8_default));
                let (default, other) = ((rows_default, int8_default), (rows_default, 1 - int8_default));
                let mut samples = std::collections::BTreeMap::<(u64, u64), Vec<f64>>::new();
                // Without the search, the formed configurations alone: the
                // default, and with `forms` its ROWS in the other form.
                if search != "full" {
                    let reported = [("default", default), ("other_form", other)].into_iter().filter(|(_, c)| points.contains_key(c));
                    for (choice, config) in reported.collect::<Vec<_>>() {
                        report(choice, config, samples.entry(config).or_default());
                    }
                    continue;
                }
                for form in [1, 0].into_iter().filter(|i| points.contains_key(&(rows_default, *i))) {
                    samples.insert((rows_default, form), vec![pass((rows_default, form))]);
                }
                let fastest = |samples: &std::collections::BTreeMap<_, Vec<f64>>| samples.values().map(|s| quantiles(s.clone())[0]).fold(f64::INFINITY, f64::min);
                let forms = samples.iter().filter(|(_, s)| s[0] <= 2.0 * fastest(&samples)).map(|(c, _)| c.1).collect::<Vec<_>>();
                for config in points.keys().filter(|c| forms.contains(&c.1) && !samples.contains_key(c)).copied().collect::<Vec<_>>() {
                    samples.insert(config, vec![pass(config)]);
                }
                let near = samples.iter().filter(|(_, s)| s[0] <= 1.25 * fastest(&samples)).map(|(c, _)| *c).collect::<Vec<_>>();
                for config in near {
                    let more = [pass(config), pass(config)];
                    samples.get_mut(&config).unwrap().extend(more);
                }
                let best = *samples.iter().min_by(|a, b| quantiles(a.1.clone())[0].total_cmp(&quantiles(b.1.clone())[0])).unwrap().0;
                // The fastest, the declared default, and the default's ROWS in
                // the other form (INT8), each once: a run settles the
                // default's choice between the forms.
                let mut reported = vec![("best", best)];
                if best != default {
                    reported.push(("default", default));
                }
                if best != other && points.contains_key(&other) {
                    reported.push(("other_form", other));
                }
                for (choice, config) in reported {
                    report(choice, config, samples.entry(config).or_default());
                }
            }
        }
    }
}
