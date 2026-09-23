//! The Metal native float conformance rows (design A8 §2.2.2, G-A8-new-7).
//!
//! `profile.rs` is the only code owner of the rows and records each row's
//! basis. These tests check that the stated table is contract §2.12.2 row for
//! row, and run the rows that one MSL operator realizes under the production
//! compile options against the language recipe. The executions are
//! regression guards over a hard corpus and random operands; they are never a
//! row's basis (sampling is not a basis, A5 §2.5).
use crate::facts::MatrixCombination;
use crate::profile::{compile_options, open_device};
use crate::test_support::metal_device;
use crate::MetalDevice;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLLibrary, MTLResourceOptions,
    MTLSize,
};
use seismic_ir::kernel::ops::SubgroupCombine;
use seismic_ir::target::{NativeFloatBehavior, NativeFloatOp};
use seismic_lang::intrinsics::MathOp;
use seismic_lang::reference_math::{
    evaluate, float_literal, scalar_recipe, ReferenceScalar, ScalarOp,
};
use seismic_lang::syntax::ast::BinaryOp;
use seismic_lang::types::DType;

const FLOATS: [DType; 3] = [DType::F32, DType::F16, DType::BF16];

/// Contract §2.12.2's Metal table, transcribed row for row. A row change is an
/// edit to the contract first (I-18), then to `profile.rs` and this copy.
fn contract_rows() -> Vec<((NativeFloatOp, DType), NativeFloatBehavior)> {
    use NativeFloatBehavior::{Deviating, Exact, FlushesSubnormals};
    use NativeFloatOp as Op;
    let mut rows = Vec::new();
    let mut row = |op: NativeFloatOp, dtype: DType, behavior| rows.push(((op, dtype), behavior));
    // F32 Add, Sub, Mul, Fma: FlushesSubnormals (Contract).
    for op in [Op::Add, Op::Sub, Op::Mul, Op::Fma] {
        row(op, DType::F32, FlushesSubnormals);
    }
    // F32 Rem: FlushesSubnormals (Contract, `fmod` 0 ulp).
    row(Op::Rem, DType::F32, FlushesSubnormals);
    // F32 Div: Deviating (I-18).
    row(Op::Div, DType::F32, Deviating);
    // F32 Sqrt: FlushesSubnormals (Exhaustive).
    row(Op::Sqrt, DType::F32, FlushesSubnormals);
    // F16 Add, Sub, Mul, Div, Sqrt: Exact (Exhaustive).
    for op in [Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Sqrt] {
        row(op, DType::F16, Exact);
    }
    // F16 Fma: Exact (Proof, round-to-odd).
    row(Op::Fma, DType::F16, Exact);
    // BF16 Add, Sub, Mul, Sqrt: FlushesSubnormals (Proof + F32 rows).
    for op in [Op::Add, Op::Sub, Op::Mul, Op::Sqrt] {
        row(op, DType::BF16, FlushesSubnormals);
    }
    // BF16 Div, BF16 Fma: Deviating (I-18, I-19).
    row(Op::Div, DType::BF16, Deviating);
    row(Op::Fma, DType::BF16, Deviating);
    // ConvertFrom(F16)->F32, ConvertFrom(F32)->F16, ConvertFrom(BF16)->F32: Exact (Exhaustive).
    row(Op::ConvertFrom(DType::F16), DType::F32, Exact);
    row(Op::ConvertFrom(DType::F32), DType::F16, Exact);
    row(Op::ConvertFrom(DType::BF16), DType::F32, Exact);
    // ConvertFrom(F32)->BF16: Exact (integer RNE helper, Proof).
    row(Op::ConvertFrom(DType::F32), DType::BF16, Exact);
    // F32/F16/BF16 Min, Max, Compare; ConvertToInteger: Exact (Proof).
    for dtype in FLOATS {
        for op in [Op::Min, Op::Max, Op::Compare, Op::ConvertToInteger] {
            row(op, dtype, Exact);
        }
    }
    // Exp, Log, Sin, Cos, Rsqrt: Deviating.
    for dtype in FLOATS {
        for op in [Op::Exp, Op::Log, Op::Sin, Op::Cos, Op::Rsqrt] {
            row(op, dtype, Deviating);
        }
    }
    // SubgroupFold(Add/Max/Min): float Deviating; I32/U32 Exact (Contract).
    for combine in [SubgroupCombine::Add, SubgroupCombine::Max, SubgroupCombine::Min] {
        for dtype in FLOATS {
            row(Op::SubgroupFold(combine), dtype, Deviating);
        }
        for dtype in [DType::I32, DType::U32] {
            row(Op::SubgroupFold(combine), dtype, Exact);
        }
    }
    // FragmentMultiplyAccumulate(F32) over F16/F32: Deviating.
    for dtype in [DType::F16, DType::F32] {
        row(Op::FragmentMultiplyAccumulate(DType::F32), dtype, Deviating);
    }
    rows
}

/// Every key the conformance vocabulary can name, so that a stated row the
/// contract lacks is found as well as a missing one.
fn every_key() -> Vec<(NativeFloatOp, DType)> {
    use NativeFloatOp as Op;
    let mut ops = vec![
        Op::Add,
        Op::Sub,
        Op::Mul,
        Op::Div,
        Op::Rem,
        Op::Min,
        Op::Max,
        Op::Fma,
        Op::Sqrt,
        Op::Rsqrt,
        Op::Exp,
        Op::Log,
        Op::Sin,
        Op::Cos,
        Op::Compare,
        Op::ConvertToInteger,
    ];
    ops.extend(DType::ALL.map(Op::ConvertFrom));
    ops.extend([SubgroupCombine::Add, SubgroupCombine::Max, SubgroupCombine::Min].map(Op::SubgroupFold));
    ops.extend(DType::ALL.map(Op::FragmentMultiplyAccumulate));
    ops.into_iter()
        .flat_map(|op| DType::ALL.map(move |dtype| (op, dtype)))
        .collect()
}

#[test]
fn stated_rows_equal_the_contract_table() {
    let expected = contract_rows();
    for (index, (key, _)) in expected.iter().enumerate() {
        assert!(
            expected[..index].iter().all(|(earlier, _)| earlier != key),
            "the transcription states {key:?} twice"
        );
    }
    let description = open_device(&metal_device()).expect("Metal device description");
    let table = description.native_float();
    let mismatches: Vec<String> = every_key()
        .into_iter()
        .filter_map(|(op, dtype)| {
            let contract = expected
                .iter()
                .find(|(key, _)| *key == (op, dtype))
                .map(|(_, behavior)| *behavior);
            let stated = table.behavior(op, dtype);
            (stated != contract)
                .then(|| format!("{op:?} {dtype:?}: profile.rs states {stated:?}, contract {contract:?}"))
        })
        .collect();
    assert!(
        mismatches.is_empty(),
        "{} row(s) differ from contract §2.12.2:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}

#[test]
fn subgroup_width_is_the_pipeline_execution_width() {
    let probe = NativeProbe::open();
    let pipeline = probe.pipeline(
        "#include <metal_stdlib>\nusing namespace metal;\n\
         kernel void probe(device uint* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] = i; }\n",
    );
    let width = u32::try_from(pipeline.threadExecutionWidth()).expect("execution width fits u32");
    assert_eq!(probe.description.limits().subgroup_width, Some(width));
    assert_eq!(width, 32);
}

/// One 8x8x8 fragment format per multiply-accumulate combination the
/// compiler accepts; the combinations include the two contract Fragment rows.
#[test]
fn fragment_formats_are_the_matrix_combinations() {
    let description = open_device(&metal_device()).expect("Metal device description");
    let combinations = &description.facts().matrix_combinations;
    for (left, accumulator) in [(DType::F16, DType::F32), (DType::F32, DType::F32)] {
        assert!(
            combinations.contains(&MatrixCombination { accumulator, left, right: left }),
            "the compiler rejects the {left:?} x {left:?} -> {accumulator:?} multiply-accumulate"
        );
    }
    let mut stated: Vec<_> = description
        .fragment_formats()
        .iter()
        .map(|format| {
            (
                (format.rows, format.columns, format.depth),
                (format.accumulator, format.left, format.right),
            )
        })
        .collect();
    stated.sort();
    let expected: Vec<_> = combinations
        .iter()
        .map(|combination| {
            ((8, 8, 8), (combination.accumulator, combination.left, combination.right))
        })
        .collect();
    assert_eq!(stated, expected);
}

/// One probe kernel per operation, compiled with the production options and
/// run over a whole operand set. Values travel as their bit patterns.
struct NativeProbe {
    device: MetalDevice,
    description: std::sync::Arc<seismic_target::DeviceDescription<crate::Metal>>,
}

/// Lanes per threadgroup; operand sets are padded to a multiple of it.
const GROUP: usize = 256;

impl NativeProbe {
    fn open() -> Self {
        let device = metal_device();
        let description = open_device(&device).expect("Metal device description");
        Self { device, description }
    }

    /// Runs `expression` (over `x`, `y`, `z` of the MSL type of `dtype`) once
    /// per operand tuple and returns the result bits.
    fn run(&self, dtype: DType, expression: &str, operands: &[Vec<u32>]) -> Vec<u32> {
        let (value, word, width) = match dtype {
            DType::F32 => ("float", "uint", 4),
            DType::F16 => ("half", "ushort", 2),
            DType::BF16 => ("bfloat", "ushort", 2),
            DType::I32 => ("int", "uint", 4),
            DType::U32 => ("uint", "uint", 4),
            other => panic!("no single-operator probe for {other:?}"),
        };
        let count = operands[0].len();
        assert!(operands.iter().all(|operand| operand.len() == count));
        let names = ["x", "y", "z"];
        let loads: String = (0..operands.len())
            .map(|i| format!("    {value} {} = as_type<{value}>(a{i}[i]);\n", names[i]))
            .collect();
        let parameters: String = (0..operands.len())
            .map(|i| format!("device const {word}* a{i} [[buffer({i})]], "))
            .collect();
        let source = format!(
            "#include <metal_stdlib>\nusing namespace metal;\n#pragma clang fp contract(off)\n\
             kernel void probe({parameters}device {word}* o [[buffer({out})]], uint i [[thread_position_in_grid]]) {{\n\
             {loads}    o[i] = as_type<{word}>({value}({expression}));\n}}\n",
            out = operands.len(),
        );
        let padded = count.div_ceil(GROUP) * GROUP;
        let inputs: Vec<Vec<u8>> = operands
            .iter()
            .map(|operand| {
                let mut bytes = vec![0u8; padded * width];
                for (slot, bits) in bytes.chunks_exact_mut(width).zip(operand) {
                    slot.copy_from_slice(&bits.to_le_bytes()[..width]);
                }
                bytes
            })
            .collect();
        let bytes = self.dispatch(&source, &inputs, padded * width, padded / GROUP, GROUP);
        bytes[..count * width]
            .chunks_exact(width)
            .map(|chunk| {
                let mut word = [0u8; 4];
                word[..width].copy_from_slice(chunk);
                u32::from_le_bytes(word)
            })
            .collect()
    }

    /// Compiles `source`'s `probe` entry, binds `inputs` at buffers
    /// `0..inputs.len()` and a zeroed output of `output_bytes` after them,
    /// dispatches `groups` threadgroups of `group` threads and returns the
    /// output bytes.
    fn dispatch(
        &self,
        source: &str,
        inputs: &[Vec<u8>],
        output_bytes: usize,
        groups: usize,
        group: usize,
    ) -> Vec<u8> {
        let pipeline = self.pipeline(source);
        let buffers: Vec<_> = inputs.iter().map(|bytes| self.buffer(bytes)).collect();
        let output = self.buffer(&vec![0u8; output_bytes]);
        let queue = self.device.queue();
        let command = queue.commandBuffer().expect("command buffer");
        let encoder = command.computeCommandEncoder().expect("compute encoder");
        encoder.setComputePipelineState(&pipeline);
        for (index, buffer) in buffers.iter().chain([&output]).enumerate() {
            unsafe { encoder.setBuffer_offset_atIndex(Some(&**buffer), 0, index) };
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: groups, height: 1, depth: 1 },
            MTLSize { width: group, height: 1, depth: 1 },
        );
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed, "{:?}", command.error());
        unsafe { std::slice::from_raw_parts(output.contents().as_ptr().cast::<u8>(), output_bytes) }
            .to_vec()
    }

    fn pipeline(&self, source: &str) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        let raw = self.device.handle().raw();
        let options = compile_options(self.description.facts().language);
        let library = raw
            .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
            .unwrap_or_else(|e| panic!("probe does not compile: {e}\n{source}"));
        let function = library
            .newFunctionWithName(&NSString::from_str("probe"))
            .expect("probe entry");
        raw.newComputePipelineStateWithFunction_error(&function)
            .expect("probe pipeline")
    }

    fn buffer(&self, bytes: &[u8]) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let buffer = self
            .device
            .handle()
            .raw()
            .newBufferWithLength_options(bytes.len().max(1), MTLResourceOptions::StorageModeShared)
            .expect("shared buffer");
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.contents().as_ptr().cast(), bytes.len())
        };
        buffer
    }
}

/// The recipe's result for every operand tuple, evaluated on all host cores
/// (a recipe evaluation costs tens of microseconds in the test profile).
fn recipe_results(op: ScalarOp, dtype: DType, operands: &[Vec<u32>]) -> Vec<u32> {
    let recipe = scalar_recipe(op, &vec![dtype; operands.len()]);
    let count = operands[0].len();
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    let chunk = count.div_ceil(threads).max(1);
    let mut results = vec![0u32; count];
    std::thread::scope(|scope| {
        for (index, slice) in results.chunks_mut(chunk).enumerate() {
            let recipe = &recipe;
            scope.spawn(move || {
                for (offset, result) in slice.iter_mut().enumerate() {
                    let at = index * chunk + offset;
                    let inputs: Vec<ReferenceScalar> = operands
                        .iter()
                        .map(|operand| ReferenceScalar::from_bits(dtype, operand[at]))
                        .collect();
                    *result = evaluate(recipe, &inputs).expect("float recipes do not fail").bits();
                }
            });
        }
    });
    results
}

struct Format {
    exponent: u32,
    mantissa: u32,
    width: u32,
}

impl Format {
    fn of(dtype: DType) -> Self {
        match dtype {
            DType::F32 => Format { exponent: 0xff, mantissa: 23, width: 32 },
            DType::F16 => Format { exponent: 0x1f, mantissa: 10, width: 16 },
            other => panic!("{other:?} is not probed"),
        }
    }
    fn exponent_field(&self, bits: u32) -> u32 {
        (bits >> self.mantissa) & self.exponent
    }
    fn fraction(&self, bits: u32) -> u32 {
        bits & ((1 << self.mantissa) - 1)
    }
    fn is_nan(&self, bits: u32) -> bool {
        self.exponent_field(bits) == self.exponent && self.fraction(bits) != 0
    }
    fn is_zero(&self, bits: u32) -> bool {
        bits & !(1 << (self.width - 1)) == 0
    }
    fn is_subnormal(&self, bits: u32) -> bool {
        self.exponent_field(bits) == 0 && self.fraction(bits) != 0
    }
    /// A subnormal becomes the zero of its sign.
    fn flush(&self, bits: u32) -> u32 {
        if self.is_subnormal(bits) {
            bits & (1 << (self.width - 1))
        } else {
            bits
        }
    }
    /// Equal payloads, or two NaNs (a NaN result's payload is unspecified).
    fn same(&self, a: u32, b: u32) -> bool {
        a == b || (self.is_nan(a) && self.is_nan(b))
    }
}

/// A5 `subnormal_hazard`: a subnormal operand, or a native result whose
/// exponent field is zero while every scaling operand is nonzero.
fn hazard(format: &Format, operands: &[u32], scaling: &[usize], native: u32) -> bool {
    operands.iter().any(|bits| format.is_subnormal(*bits))
        || (format.exponent_field(native) == 0
            && scaling.iter().all(|index| !format.is_zero(operands[*index])))
}

/// Deterministic operand bits (SplitMix64).
struct Bits(u64);

impl Bits {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        (z ^ (z >> 31)) as u32
    }
}

const F32_SPECIALS: [u32; 14] = [
    0x0000_0000, 0x8000_0000, 0x3f80_0000, 0xbf80_0000, 0x7f80_0000, 0xff80_0000, 0x7fc0_0000,
    0x7f7f_ffff, 0x0080_0000, 0x8080_0000, 0x0000_0001, 0x807f_ffff, 0x3f80_0001, 0x4b80_0000,
];

/// F32 operand tuples whose exact results sit at or next to rounding
/// midpoints, plus special values and `random` uniform bit patterns.
fn f32_corpus(arity: usize, op: &str, random: usize, seed: u64) -> Vec<Vec<u32>> {
    let mut bits = Bits(seed);
    let mut tuples: Vec<Vec<u32>> = Vec::new();
    for a in F32_SPECIALS {
        for b in F32_SPECIALS {
            tuples.push(vec![a, b, F32_SPECIALS[(a ^ b) as usize % F32_SPECIALS.len()]]);
        }
    }
    for _ in 0..4096 {
        // A normal value with a random mantissa and a moderate exponent.
        let a = (bits.next() & 0x807f_ffff) | ((bits.next() % 200 + 27) << 23);
        let exponent = (a >> 23) & 0xff;
        let hard = match op {
            // Half an ulp of `a` (a tie), three halves, and just under half.
            "add" | "sub" => {
                let half = (exponent - 24) << 23;
                [half, half | 0x0040_0000, (exponent - 25) << 23 | 0x007f_ffff]
                    [bits.next() as usize % 3]
                    | (bits.next() & 0x8000_0000)
            }
            // (1 + k 2^-12)^2 carries k^2 2^-24: ties and near-ties.
            "mul" | "fma" => {
                let k = bits.next() & 0xfff | 1;
                0x3f80_0000 | (k << 11)
            }
            // A divisor far below the dividend: the remainder keeps many bits.
            "rem" => (bits.next() & 0x807f_ffff) | ((exponent.saturating_sub(bits.next() % 40 + 1)).max(1) << 23),
            _ => bits.next(),
        };
        let first = if matches!(op, "mul" | "fma") { hard } else { a };
        // fma: c cancels the rounded product, leaving the exact residual.
        let c = if op == "fma" {
            (f32::from_bits(first) * f32::from_bits(hard)).to_bits() ^ 0x8000_0000
        } else {
            bits.next()
        };
        tuples.push(vec![first, hard, c]);
    }
    if op == "sqrt" {
        for _ in 0..4096 {
            // Near perfect squares: the root is a rounding boundary.
            let y = f32::from_bits((bits.next() & 0x007f_ffff) | ((bits.next() % 60 + 97) << 23));
            let square = (y * y).to_bits();
            let neighbour = square.wrapping_add(bits.next() % 3).wrapping_sub(1);
            tuples.push(vec![neighbour, 0, 0]);
        }
    }
    for _ in 0..random {
        tuples.push(vec![bits.next(), bits.next(), bits.next()]);
    }
    (0..arity)
        .map(|operand| tuples.iter().map(|tuple| tuple[operand]).collect())
        .collect()
}

/// Checks one `FlushesSubnormals` row: off the hazard the native result is
/// the recipe's bit for bit; on it, the native result is the recipe's result
/// with subnormal operands and result flushed, or the recipe's own result.
fn check_flushes_subnormals(
    probe: &NativeProbe,
    name: &str,
    op: ScalarOp,
    expression: &str,
    operands: &[Vec<u32>],
    scaling: &[usize],
) {
    let format = Format::of(DType::F32);
    let native = probe.run(DType::F32, expression, operands);
    let recipe = recipe_results(op, DType::F32, operands);
    let flushed_operands: Vec<Vec<u32>> = operands
        .iter()
        .map(|operand| operand.iter().map(|bits| format.flush(*bits)).collect())
        .collect();
    let flushed = recipe_results(op, DType::F32, &flushed_operands);
    let mut failures = Vec::new();
    for i in 0..native.len() {
        let tuple: Vec<u32> = operands.iter().map(|operand| operand[i]).collect();
        let accepted = if hazard(&format, &tuple, scaling, native[i]) {
            format.same(native[i], recipe[i]) || format.same(native[i], format.flush(flushed[i]))
        } else {
            format.same(native[i], recipe[i])
        };
        if !accepted {
            failures.push(format!(
                "{tuple:08x?}: native {:08x}, recipe {:08x}",
                native[i], recipe[i]
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "F32 {name}: {} of {} results contradict FlushesSubnormals:\n{}",
        failures.len(),
        native.len(),
        failures[..failures.len().min(16)].join("\n")
    );
}

/// 2^16 uniform operand tuples per row: the recipe costs tens of microseconds
/// per evaluation in the test profile, so this is the regression-guard size.
const RANDOM: usize = 1 << 16;

#[test]
fn f32_multiply_flushes_a_subnormal_product() {
    let probe = NativeProbe::open();
    let product = probe.run(DType::F32, "x * y", &[vec![0x0000_0001], vec![0x3f80_0000]]);
    assert_eq!(product, [0x0000_0000], "the F32 Mul row would be Exact");
}

#[test]
fn f32_flushes_subnormals_rows_agree_with_the_recipe_off_the_hazard() {
    let probe = NativeProbe::open();
    let rows: [(&str, ScalarOp, &str, usize, &[usize]); 6] = [
        ("add", ScalarOp::Binary(BinaryOp::Add), "x + y", 2, &[0, 1]),
        ("sub", ScalarOp::Binary(BinaryOp::Sub), "x - y", 2, &[0, 1]),
        ("mul", ScalarOp::Binary(BinaryOp::Mul), "x * y", 2, &[0, 1]),
        ("fma", ScalarOp::Math(MathOp::Fma), "fma(x, y, z)", 3, &[0, 1]),
        ("rem", ScalarOp::Binary(BinaryOp::Rem), "fmod(x, y)", 2, &[0]),
        ("sqrt", ScalarOp::Math(MathOp::Sqrt), "sqrt(x)", 1, &[]),
    ];
    for (seed, (name, op, expression, arity, scaling)) in rows.into_iter().enumerate() {
        let operands = f32_corpus(arity, name, RANDOM, seed as u64);
        check_flushes_subnormals(&probe, name, op, expression, &operands, scaling);
    }
}

#[test]
fn f16_exact_rows_equal_the_recipe() {
    let probe = NativeProbe::open();
    let format = Format::of(DType::F16);
    let every: Vec<u32> = (0..1u32 << 16).collect();
    let mut bits = Bits(16);
    let pairs: [Vec<u32>; 2] = std::array::from_fn(|_| (0..RANDOM).map(|_| bits.next() & 0xffff).collect());
    let rows: [(&str, ScalarOp, &str, Vec<Vec<u32>>); 5] = [
        ("add", ScalarOp::Binary(BinaryOp::Add), "x + y", pairs.to_vec()),
        ("sub", ScalarOp::Binary(BinaryOp::Sub), "x - y", pairs.to_vec()),
        ("mul", ScalarOp::Binary(BinaryOp::Mul), "x * y", pairs.to_vec()),
        ("div", ScalarOp::Binary(BinaryOp::Div), "x / y", pairs.to_vec()),
        // Every F16 input.
        ("sqrt", ScalarOp::Math(MathOp::Sqrt), "sqrt(x)", vec![every]),
    ];
    for (name, op, expression, operands) in rows {
        let native = probe.run(DType::F16, expression, &operands);
        let recipe = recipe_results(op, DType::F16, &operands);
        let failures: Vec<String> = (0..native.len())
            .filter(|i| !format.same(native[*i], recipe[*i]))
            .map(|i| {
                let tuple: Vec<u32> = operands.iter().map(|operand| operand[i]).collect();
                format!("{tuple:04x?}: native {:04x}, recipe {:04x}", native[i], recipe[i])
            })
            .collect();
        assert!(
            failures.is_empty(),
            "F16 {name}: {} of {} results differ from the recipe:\n{}",
            failures.len(),
            native.len(),
            failures[..failures.len().min(16)].join("\n")
        );
    }
}

/// `Deviating` rows are admitted only where floating deviation is recorded;
/// they must compile under the production options and run. These are the
/// rows one MSL operator realizes; the BF16 Div, BF16 Fma and BF16
/// transcendental rows are realized through the renderer's float widening and
/// run in A8-L5's render tests.
#[test]
fn deviating_rows_compile_and_run() {
    let probe = NativeProbe::open();
    let operands = [vec![0x3f80_0000, 0x4049_0fdb, 0x0000_0001, 0x7fc0_0000], vec![0x4000_0000; 4]];
    for expression in ["x / y", "exp(x)", "log(x)", "sin(x)", "cos(x)", "rsqrt(x)"] {
        let arity = if expression == "x / y" { 2 } else { 1 };
        assert_eq!(probe.run(DType::F32, expression, &operands[..arity]).len(), 4);
    }
    let half = [vec![0x3c00, 0x4248, 0x0001, 0x7e00], vec![0x4000; 4]];
    for expression in ["exp(x)", "log(x)", "sin(x)", "cos(x)", "rsqrt(x)"] {
        assert_eq!(probe.run(DType::F16, expression, &half[..1]).len(), 4);
    }
}

/// The float `SubgroupFold` rows are `Deviating`: each is one MSL collective
/// on the element type, which must compile and run. Lane `i` holds `i`; max
/// and min are exact in any order, and so is the sum where every partial sum
/// is representable (every subset sum of `0..64` is below 2^11, exact in F32
/// and F16 but not in BF16's 8-bit significand).
#[test]
fn float_subgroup_folds_compile_and_run() {
    let probe = NativeProbe::open();
    let collectives = &probe.description.facts().scalar_collective_dtypes;
    assert!(collectives.contains(&DType::F32) && collectives.contains(&DType::F16));
    for dtype in FLOATS.into_iter().filter(|dtype| collectives.contains(dtype)) {
        let bits_of = |value: u32| float_literal(dtype, f64::from(value)).bits();
        let lanes: Vec<u32> = (0..64).map(bits_of).collect();
        for fold in ["simd_sum", "simd_max", "simd_min"] {
            let native = probe.run(dtype, &format!("{fold}(x)"), &[lanes.clone()]);
            assert_eq!(native.len(), 64);
            if fold == "simd_sum" && dtype == DType::BF16 {
                continue;
            }
            let first = |group: u32| group * 32;
            let expected = |group: u32| match fold {
                "simd_sum" => (first(group)..first(group) + 32).sum::<u32>(),
                "simd_max" => first(group) + 31,
                _ => first(group),
            };
            for group in 0..2u32 {
                let range = first(group) as usize..first(group) as usize + 32;
                assert!(
                    native[range].iter().all(|lane| *lane == bits_of(expected(group))),
                    "{dtype:?} {fold} over subgroup {group} is not {}",
                    expected(group)
                );
            }
        }
    }
}

/// The `FragmentMultiplyAccumulate(F32)` rows over F16 and F32 are
/// `Deviating`: each is one `simdgroup_multiply_accumulate` on 8x8 fragments,
/// which must compile and run. Small integer operands make every product and
/// partial sum exact, so `D = A B + C` holds in any accumulation order.
#[test]
fn fragment_multiply_accumulate_compiles_and_runs() {
    let probe = NativeProbe::open();
    let a = |row: usize, k: usize| ((row + k) % 5) as f64;
    let b = |k: usize, column: usize| ((3 * k + column) % 4) as f64;
    let c = |row: usize, column: usize| row as f64 - column as f64;
    for (operand, msl) in [(DType::F16, "half"), (DType::F32, "float")] {
        let source = format!(
            "#include <metal_stdlib>\nusing namespace metal;\n\
             kernel void probe(device const {msl}* a [[buffer(0)]], device const {msl}* b [[buffer(1)]], \
             device const float* c [[buffer(2)]], device float* d [[buffer(3)]]) {{\n\
             simdgroup_matrix<{msl}, 8, 8> af; simdgroup_matrix<{msl}, 8, 8> bf;\n\
             simdgroup_matrix<float, 8, 8> cf; simdgroup_matrix<float, 8, 8> df;\n\
             simdgroup_load(af, a, 8); simdgroup_load(bf, b, 8); simdgroup_load(cf, c, 8);\n\
             simdgroup_multiply_accumulate(df, af, bf, cf);\n\
             simdgroup_store(df, d, 8);\n}}\n"
        );
        let matrix = |dtype: DType, entry: &dyn Fn(usize, usize) -> f64| -> Vec<u8> {
            let width = if dtype == DType::F16 { 2 } else { 4 };
            (0..64)
                .flat_map(|index| {
                    let bits = float_literal(dtype, entry(index / 8, index % 8)).bits();
                    bits.to_le_bytes()[..width].to_vec()
                })
                .collect()
        };
        let inputs = [matrix(operand, &a), matrix(operand, &b), matrix(DType::F32, &c)];
        let bytes = probe.dispatch(&source, &inputs, 64 * 4, 1, 32);
        let native: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("4 bytes")))
            .collect();
        for (index, value) in native.iter().enumerate() {
            let (row, column) = (index / 8, index % 8);
            let expected = (0..8).map(|k| a(row, k) * b(k, column)).sum::<f64>() + c(row, column);
            assert_eq!(
                f64::from(*value),
                expected,
                "{operand:?} fragment multiply-accumulate at ({row}, {column})"
            );
        }
    }
}

/// I32/U32 subgroup folds are `Exact` (Contract): wrapping add, max and min
/// over each 32-lane subgroup, signed for I32 and unsigned for U32.
#[test]
fn integer_subgroup_folds_are_exact() {
    let probe = NativeProbe::open();
    let mut bits = Bits(32);
    let lanes: Vec<u32> = (0..256).map(|_| bits.next()).collect();
    let folds: [(&str, fn(u32, u32) -> u32, fn(u32, u32) -> u32); 3] = [
        ("simd_sum", u32::wrapping_add, |a, b| (a as i32).wrapping_add(b as i32) as u32),
        ("simd_max", u32::max, |a, b| (a as i32).max(b as i32) as u32),
        ("simd_min", u32::min, |a, b| (a as i32).min(b as i32) as u32),
    ];
    for (fold, unsigned, signed) in folds {
        for (dtype, combine) in [(DType::U32, unsigned), (DType::I32, signed)] {
            let native = probe.run(dtype, &format!("{fold}(x)"), &[lanes.clone()]);
            for (group, chunk) in lanes.chunks(32).enumerate() {
                let expected = chunk.iter().copied().reduce(combine).expect("32 lanes");
                assert!(
                    native[group * 32..group * 32 + 32].iter().all(|lane| *lane == expected),
                    "{fold} over subgroup {group} is not the exact {dtype:?} fold"
                );
            }
        }
    }
}
