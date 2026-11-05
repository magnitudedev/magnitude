//! The backend's probes (`seismic_lang::device::vulkan`, its `probes`):
//! kernels that establish what no query reports, or check that a reported
//! feature works where a driver has been seen to get it wrong, each with
//! the result expected of it written beside it.
//!
//! The measurement job runs each probe once per device and driver version,
//! in a probe process of its own (`seismic_runtime::probes`), which opens
//! the device ([`open`]) and runs the probe ([`run`]), since a probe can
//! crash the driver's compiler; it stores the result, which later opens
//! read ([`crate::Device::open`]). Opening a device runs no probe, and a
//! load never does.

use crate::device::Device;
use crate::direct::{DirectBatch, DirectLaunch};
use crate::formation::{self, DirectModule};
use crate::memory::Buffer;
use crate::seal::Environment;
use seismic_lang::device::ProbeResult;
use seismic_native_target::{ProgramEntry, StatisticValue};

/// The revision of these probes: their kernels and expected results. It
/// keys their stored results, so a change runs them again.
pub const REVISION: u32 = 1;

/// What every probe kernel declares: the floor's extensions it uses and the
/// argument block.
const HEAD: &str = include_str!("probe/head.glsl");
/// The extensions `wide_accumulators` adds, declared before [`HEAD`].
const WIDE_EXTENSIONS: &str = "#extension GL_EXT_shared_memory_block : require\n\
     #extension GL_KHR_memory_scope_semantics : require\n\
     #extension GL_KHR_shader_subgroup_basic : require\n\
     #extension GL_KHR_shader_subgroup_vote : require\n\
     #extension GL_KHR_shader_subgroup_shuffle : require\n\
     #extension GL_KHR_cooperative_matrix : require\n";

/// Run probe `name` on `device`: its answer checked against what is
/// expected of it (a `Support` probe answers 0 or 1), or why it cannot
/// answer on this device. `Err` is a failure that says nothing of the
/// device (memory, submission), so nothing is to be kept of it.
pub fn run(device: &Device, name: &str) -> Result<ProbeResult<u64>, String> {
    match name {
        "rte_32_division" => rte_32_division(device),
        "wide_accumulators" => wide_accumulators(device),
        "thread_registers" => Ok(registers(device, 32)),
        // The registers of each of 1024 threads, which share the file.
        "register_file" => Ok(match registers(device, 1024) {
            ProbeResult::Answer(registers) => ProbeResult::Answer(registers * 1024),
            unanswerable => unanswerable,
        }),
        other => Err(format!("`{other}` is not a Vulkan probe")),
    }
}

/// The pipeline of `body` (after the head and `extensions`, with `defines`)
/// sealed for `environment`, one workgroup of `threads`. The driver's
/// compiler rejecting it is the probe's finding, not an error: it rejected
/// what it reports supporting. The host's compilation failing is the
/// probe's own bug, which says nothing of the device: it panics.
fn form(
    device: &Device,
    defines: &str,
    extensions: &str,
    body: &str,
    environment: Environment,
    threads: u32,
) -> Result<DirectModule, String> {
    let source = format!("#version 460\n{defines}{extensions}{HEAD}{body}");
    let words = formation::compile(&source, "main", environment)
        .unwrap_or_else(|error| panic!("a probe kernel does not compile on the host: {error}\n{source}"));
    let entry = ProgramEntry { symbol: "main".into(), group_size: Some([threads, 1, 1]), constants: Vec::new() };
    DirectModule::compile(device, &[(&words[..], &entry)]).map(|(module, _)| module).map_err(|error| error.to_string())
}

/// Run `module`'s one pipeline over `groups` workgroups with `buffers` and
/// `words`, and wait for it.
fn dispatch(device: &Device, module: &DirectModule, buffers: &[(&Buffer, u64)], words: &[u8], groups: u64) -> Result<(), String> {
    let mut batch = DirectBatch::new(device).map_err(|error| error.to_string())?;
    batch
        .launch(&DirectLaunch { module, function: 0, buffers, words, scalar_results: None, groups: [groups, 1, 1], grid: None })
        .map_err(|error| error.to_string())?;
    batch.commit().and_then(|submission| submission.finish()).map_err(|error| error.to_string())
}

/// A storage buffer holding `bytes`, then `extra` zero bytes.
fn upload(device: &Device, bytes: &[u8], extra: usize) -> Result<Buffer, String> {
    let buffer = device.allocate((bytes.len() + extra) as u64, 256, None).map_err(|error| error.to_string())?;
    let mut contents = bytes.to_vec();
    contents.resize(bytes.len() + extra, 0);
    device.write(&buffer, 0, &contents).map_err(|error| error.to_string())?;
    Ok(buffer)
}

fn download(device: &Device, buffer: &Buffer, offset: u64, bytes: usize) -> Result<Vec<u8>, String> {
    let mut out = vec![0u8; bytes];
    device.read(buffer, offset, &mut out).map_err(|error| error.to_string())?;
    Ok(out)
}

/// Deterministic operand bits (xorshift).
struct Operands(u64);

impl Operands {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A normal fp32 of exponent within ±60 and any sign and mantissa, so
    /// every sum, product and quotient of two is normal.
    fn normal(&mut self) -> f32 {
        let word = self.next();
        let exponent = 127 - 60 + (word >> 32) as u32 % 121;
        f32::from_bits((word as u32 & 0x807f_ffff) | (exponent << 23))
    }
}

/// Distance in units in the last place between two finite fp32 of one sign.
fn ulps(expected: f32, actual: f32) -> u64 {
    if expected.is_sign_negative() != actual.is_sign_negative() {
        return u64::MAX;
    }
    u64::from(expected.to_bits().abs_diff(actual.to_bits()))
}

/// `rte_32_division`: a kernel of fp32 add, multiply and divide sealed with
/// `RoundingModeRTE 32` forms, runs and rounds as declared.
///
/// Expected: every sum and product equals the host's correctly rounded
/// (round to nearest, ties to even) result bit for bit, as Vulkan requires
/// of add and multiply; every quotient is within 2.5 ULP of the correctly
/// rounded one, Vulkan's precision for `OpFDiv`. 4096 operand pairs, normal
/// with normal results; about half of their sums round differently toward
/// zero. A driver whose compiler fails on the mode (NVIDIA 580.159.03 traps
/// on fp32 `OpFDiv` under it) ends the probe's process, which the
/// measurement job keeps as unanswerable: the mode is not declared.
fn rte_32_division(device: &Device) -> Result<ProbeResult<u64>, String> {
    const COUNT: usize = 4096;
    let environment = Environment { rounding_rte_32: true, ..device.formation().environment() };
    let module = match form(device, "", "", include_str!("probe/rte_32_division.glsl"), environment, 64) {
        Ok(module) => module,
        Err(reason) => {
            eprintln!("rte_32_division does not form: {reason}");
            return Ok(ProbeResult::Answer(u64::from(false)));
        }
    };
    let mut operands = Operands(0x9e37_79b9_7f4a_7c15);
    let pairs = (0..COUNT).map(|_| (operands.normal(), operands.normal())).collect::<Vec<_>>();
    let mut bytes = (COUNT as u32).to_le_bytes().to_vec();
    for (a, b) in &pairs {
        bytes.extend_from_slice(&a.to_le_bytes());
        bytes.extend_from_slice(&b.to_le_bytes());
    }
    let data = upload(device, &bytes, COUNT * 12)?;
    dispatch(device, &module, &[(&data, 0)], &[], COUNT.div_ceil(64) as u64)?;
    let results = download(device, &data, bytes.len() as u64, COUNT * 12)?;
    let value = |index: usize| f32::from_le_bytes(results[4 * index..4 * index + 4].try_into().unwrap());
    let mut mismatches = 0;
    for (index, (a, b)) in pairs.iter().enumerate() {
        let (sum, product, quotient) = (value(3 * index), value(3 * index + 1), value(3 * index + 2));
        let held = sum.to_bits() == (a + b).to_bits()
            && product.to_bits() == (a * b).to_bits()
            && 2 * ulps(a / b, quotient) <= 5;
        if !held {
            if mismatches < 4 {
                eprintln!("rte_32_division: {a:e} {b:e}: sum {sum:e} (expected {:e}), product {product:e} ({:e}), quotient {quotient:e} ({:e})", a + b, a * b, a / b);
            }
            mismatches += 1;
        }
    }
    if mismatches > 0 {
        eprintln!("rte_32_division: {mismatches} of {COUNT} pairs rounded otherwise than declared");
    }
    Ok(ProbeResult::Answer(u64::from(mismatches == 0)))
}

/// f16 bits of `value`, which is exactly representable as a normal f16 or
/// zero (`wide_accumulators`' operands are).
fn f16_bits(value: f32) -> u16 {
    if value == 0.0 {
        return 0;
    }
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    debug_assert!((1..31).contains(&exponent) && bits & 0x1fff == 0, "{value} is a normal f16");
    (sign | ((exponent as u32) << 10) | ((bits >> 13) & 0x3ff)) as u16
}

/// `wide_accumulators`: the flash-attention body's cooperative-matrix path
/// with 8 and 16 output fragments computes attention correctly.
///
/// Expected: for 16 query rows and 80 keys (two whole tiles and a partial
/// one) of operands k / 512 (|k| <= 256, exact in f16), at head widths 128
/// (8 fragments) and 256 (16 fragments), every output is within 4e-3 of the
/// host's attention over the same operands in f64; the kernel rounds only
/// the probabilities to f16 (2^-11 relative), and outputs lie in [-0.5,
/// 0.5]. Wrong outputs of a mishandled accumulator array are far outside
/// that (NVIDIA's 580 compiler was seen returning them for 16-fragment, and
/// 8-declared-2-used, arrays in an earlier form of the body; GB10 on
/// 580.159.03 computes this one within 2e-5 at 4, 8 and 16 fragments). The same
/// kernel at width 64 (4 fragments) is the control: where it fails, the probe, not the accumulators, is
/// wrong, and nothing is answered.
fn wide_accumulators(device: &Device) -> Result<ProbeResult<u64>, String> {
    const ROWS: usize = 16;
    const KEYS: usize = 80;
    let target = device.formation();
    if !(device.properties().matrix.holds() && target.float16 && target.subgroup_lanes == crate::SUBGROUP_WIDTH) {
        return Ok(ProbeResult::Unanswerable(
            "the cooperative-matrix path needs the f16 16x16x16 shape, fp16 arithmetic and 32-lane subgroups".into(),
        ));
    }
    let mut operands = Operands(0x2545_f491_4f6c_dd1d);
    let mut correct = |width: usize| -> Result<Result<bool, String>, String> {
        let mut operand = || (operands.next() % 513) as f32 / 512.0 - 0.5;
        let q = (0..ROWS * width).map(|_| operand()).collect::<Vec<_>>();
        let k = (0..KEYS * width).map(|_| operand()).collect::<Vec<_>>();
        let v = (0..KEYS * width).map(|_| operand()).collect::<Vec<_>>();
        let module = match form(
            device,
            &format!("#define W {width}u\n"),
            WIDE_EXTENSIONS,
            include_str!("probe/wide_accumulators.glsl"),
            target.environment(),
            32,
        ) {
            Ok(module) => module,
            Err(reason) => return Ok(Err(format!("width {width} does not form: {reason}"))),
        };
        let halves = |values: &[f32]| values.iter().flat_map(|value| f16_bits(*value).to_le_bytes()).collect::<Vec<_>>();
        let (q_buffer, k_buffer, v_buffer) = (upload(device, &halves(&q), 0)?, upload(device, &halves(&k), 0)?, upload(device, &halves(&v), 0)?);
        let out = upload(device, &[], ROWS * width * 4)?;
        let buffers = [(&q_buffer, 0), (&k_buffer, 0), (&v_buffer, 0), (&out, 0)];
        dispatch(device, &module, &buffers, &(KEYS as u64).to_le_bytes(), 1)?;
        let outputs = download(device, &out, 0, ROWS * width * 4)?;
        let mut worst = 0f64;
        for row in 0..ROWS {
            let scores = (0..KEYS)
                .map(|key| (0..width).map(|d| f64::from(q[row * width + d]) * f64::from(k[key * width + d])).sum::<f64>() / (width as f64).sqrt())
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights = scores.iter().map(|score| (score - maximum).exp()).collect::<Vec<_>>();
            let total = weights.iter().sum::<f64>();
            for column in 0..width {
                let expected = (0..KEYS).map(|key| weights[key] * f64::from(v[key * width + column])).sum::<f64>() / total;
                let index = 4 * (row * width + column);
                let actual = f32::from_le_bytes(outputs[index..index + 4].try_into().unwrap());
                let error = (f64::from(actual) - expected).abs();
                worst = if error.is_nan() { f64::INFINITY } else { worst.max(error) };
            }
        }
        eprintln!("wide_accumulators: width {width} ({} fragments): largest error {worst:e}", width / 16);
        Ok(Ok(worst <= 4e-3))
    };
    match correct(64)? {
        Ok(true) => {}
        Ok(false) => return Ok(ProbeResult::Unanswerable("the 4-fragment control is wrong: the probe does not hold here".into())),
        Err(reason) => return Ok(ProbeResult::Unanswerable(format!("the 4-fragment control: {reason}"))),
    }
    let mut wide = true;
    for width in [128, 256] {
        match correct(width)? {
            Ok(held) => wide &= held,
            Err(reason) => {
                eprintln!("wide_accumulators: {reason}");
                wide = false;
            }
        }
    }
    Ok(ProbeResult::Answer(u64::from(wide)))
}

/// The "Register Count" a register-heavy kernel of `threads` per workgroup
/// is formed with.
///
/// Expected: the kernel keeps 320 floats live, more than any thread's
/// registers hold (each the next step of one dependent chain, all reduced
/// together every round, so none is cheaper to recompute and no narrow
/// schedule keeps fewer), so the compiler gives it every register its workgroup
/// size allows: the per-thread ceiling at 32 threads, the register file's
/// share of each thread at 1024. The answer is the most that kernel or the
/// same keeping 640 live is given: a compiler under such pressure may choose
/// fewer and spill (NVIDIA 615.71.09 gives the 640-live kernel 168 of its
/// 255 at 32 threads), never more than the bound. At 1024 threads it is
/// below the 32-thread answer, or the file did not bound it. The NVIDIA proprietary
/// driver names the statistic "Register Count"; a driver reporting no such
/// statistic cannot answer.
fn registers(device: &Device, threads: u32) -> ProbeResult<u64> {
    if !device.facts().pipeline_statistics {
        return ProbeResult::Unanswerable("the device reports no pipeline executable statistics".into());
    }
    // A kernel that does not form answers no count.
    let count = |live: u32, threads: u32| -> Option<u64> {
        let module = form(
            device,
            &format!("#define LIVE {live}u\n"),
            "",
            include_str!("probe/registers.glsl"),
            device.formation().environment(),
            threads,
        )
        .inspect_err(|reason| eprintln!("registers: {live} live at {threads} threads does not form: {reason}"))
        .ok()?;
        let statistic = module.statistics(0).iter().find(|statistic| statistic.name == "Register Count")?;
        match statistic.value {
            StatisticValue::Uint(value) => Some(value),
            StatisticValue::Int(value) => u64::try_from(value).ok(),
            _ => None,
        }
    };
    let unanswerable = |reason: String| ProbeResult::Unanswerable(reason);
    let no_count = || unanswerable("the kernel forms with no \"Register Count\" statistic".into());
    let most = |threads: u32| {
        let (heavy, heavier) = (count(320, threads)?, count(640, threads)?);
        eprintln!("registers: {threads} threads: {heavy} registers keeping 320 live, {heavier} keeping 640");
        Some(heavy.max(heavier))
    };
    let Some(registers) = most(threads) else {
        return no_count();
    };
    if threads > 32 {
        let Some(ceiling) = most(32) else {
            return no_count();
        };
        if registers >= ceiling {
            return unanswerable(format!("{threads} threads are given the per-thread ceiling ({registers}): the register file did not bound them"));
        }
    }
    ProbeResult::Answer(registers)
}

/// The device argument of a probe process for the device of `uuid`, which
/// [`open`] reads: its 32 hex digits.
pub fn argument(uuid: &[u8; 16]) -> String {
    crate::facts::hex(uuid)
}

/// The device a probe process runs its probe on: the one of UUID `device`
/// ([`argument`]), opened with what the probes need and no stored probe
/// results. `Err` says nothing of the device's probes.
pub fn open(device: &str) -> Result<Device, String> {
    let uuid = (0..16)
        .map(|byte| device.get(2 * byte..2 * byte + 2).and_then(|hex| u8::from_str_radix(hex, 16).ok()))
        .collect::<Option<Vec<u8>>>()
        .filter(|_| device.len() == 32)
        .ok_or_else(|| format!("`{device}` is not a device UUID of 32 hex digits"))?;
    let uuid: [u8; 16] = uuid.try_into().expect("16 bytes");
    let description = crate::discover()
        .map_err(|error| error.to_string())?
        .into_iter()
        .find(|description| description.facts.uuid == uuid)
        .ok_or_else(|| format!("no Vulkan device has UUID {device}"))?;
    Device::open(uuid, crate::unbounded_accounts(&description.facts), &Default::default()).map_err(|error| error.to_string())
}
