//! MSL rendering of one typed `Kernel<Metal>` (spec §11.2).
//!
//! The renderer is a total, exhaustive match over the closed op
//! vocabulary. It makes no allocation, geometry, synchronization,
//! algorithm or precision decision: geometry and bindings come from the
//! launch and storage from the interface and the kernel's locals. Safety
//! preflights are separate schedule kernels and checks; they never become an
//! inline native op that could continue after failure. Reference math is
//! expanded into ordinary primitive IR before native emission, and Metal
//! preserves each primitive rounding boundary with fast math and contraction
//! disabled. Only explicitly approximate math remains a native math op.
//!
//! Kernel argument table (indices agreed with `compile`/`executor`):
//! `[[buffer(i)]]` for binding slot `i`, then the parameter word block,
//! then the side block (status word + result slots), then one
//! `[[threadgroup(k)]]` pointer per workgroup local.

use crate::intrinsic::MetalIntrinsic;
use crate::Metal;
use seismic_compiler::kernel::ops::{
    BarrierScope, BinaryOp, BitOp, Block, ClosedOpView, ClosedPlace, ClosedPlaceKind, ClosedValue,
    CmpOp, ConstantValue, ErasedValue, GeometryValue, LogicOp, LogicalSliceAxis, LogicalTensorMap,
    LogicalViewStep, PlaceRef, UnaryOp, ValueType,
};
use seismic_compiler::kernel::{BlockId, Kernel};
use seismic_compiler::storage::LaunchLocalKind;
use seismic_compiler::target::KernelEmissionLayout;
use seismic_lang::intrinsics::{AtomicOp, MathOp, ReduceOp};
use seismic_lang::registry::{
    CodeInterpretation, DecodeStep, FloatCodeFormat, PlaneEncoding, PlaneRepackRecipe, RepackExpr,
    RepresentationKind,
};
use seismic_lang::types::DType;
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// The library prelude every kernel shares (fast math contract).
pub(crate) const LIBRARY_PRELUDE: &str = r#"#include <metal_stdlib>
#pragma clang fp contract(off)
using namespace metal;
float f32_mul(float, float);
inline float seismic_decode_float_code(uint raw, uint sign_shift, uint exponent_bits,
                                        uint mantissa_bits, uint exponent_bias,
                                        float subnormal_scale, bool unsigned_code,
                                        bool finite_nan) {
  uint sign = unsigned_code ? 0u : ((raw >> sign_shift) << 31);
  uint exponent = (raw >> mantissa_bits) & ((1u << exponent_bits) - 1u);
  uint mantissa = raw & ((1u << mantissa_bits) - 1u);
  uint normal = (mantissa << (23u - mantissa_bits)) | ((exponent + exponent_bias) << 23u);
  uint subnormal = as_type<uint>(f32_mul(float(mantissa), subnormal_scale));
  uint magnitude = exponent == 0u ? subnormal : normal;
  if (finite_nan && exponent == ((1u << exponent_bits) - 1u) &&
      mantissa == ((1u << mantissa_bits) - 1u)) magnitude = 0x7fc00000u;
  return as_type<float>(magnitude ^ sign);
}
inline float seismic_decode_e2m1(uint raw) {
  return seismic_decode_float_code(raw, 3u, 2u, 1u, 126u, 0.5f, false, false);
}
inline float seismic_decode_e4m3(uint raw) {
  return seismic_decode_float_code(raw, 7u, 4u, 3u, 120u, 0.001953125f, false, true);
}
inline float seismic_decode_ue4m3(uint raw) {
  return seismic_decode_float_code(raw, 7u, 4u, 3u, 120u, 0.001953125f, true, true);
}
inline bool seismic_f32_nan(float value) {
  return (as_type<uint>(value) & 0x7fffffffu) > 0x7f800000u;
}
inline bool seismic_f32_zero(float value) {
  return (as_type<uint>(value) & 0x7fffffffu) == 0u;
}
inline bool f32_eq(float a, float b) {
  return !seismic_f32_nan(a) && !seismic_f32_nan(b) &&
         ((seismic_f32_zero(a) && seismic_f32_zero(b)) || as_type<uint>(a) == as_type<uint>(b));
}
inline bool f32_lt(float a, float b) {
  uint ai = as_type<uint>(a), bi = as_type<uint>(b);
  if (seismic_f32_nan(a) || seismic_f32_nan(b) ||
      (seismic_f32_zero(a) && seismic_f32_zero(b))) return false;
  bool as = (ai >> 31) != 0u, bs = (bi >> 31) != 0u;
  return as != bs ? as : (as ? ai > bi : ai < bi);
}
inline float f32_min(float a, float b) {
  if (seismic_f32_nan(a)) return b;
  if (seismic_f32_nan(b)) return a;
  if (seismic_f32_zero(a) && seismic_f32_zero(b))
    return as_type<float>(as_type<uint>(a) | as_type<uint>(b));
  return f32_lt(b, a) ? b : a;
}
inline float f32_max(float a, float b) {
  if (seismic_f32_nan(a)) return b;
  if (seismic_f32_nan(b)) return a;
  if (seismic_f32_zero(a) && seismic_f32_zero(b))
    return as_type<float>(as_type<uint>(a) & as_type<uint>(b));
  return f32_lt(a, b) ? b : a;
}
inline bool seismic_f16_nan(half value) {
  return (as_type<ushort>(value) & 0x7fffu) > 0x7c00u;
}
inline bool seismic_f16_zero(half value) {
  return (as_type<ushort>(value) & 0x7fffu) == 0u;
}
inline bool f16_eq(half a, half b) {
  return !seismic_f16_nan(a) && !seismic_f16_nan(b) &&
         ((seismic_f16_zero(a) && seismic_f16_zero(b)) || as_type<ushort>(a) == as_type<ushort>(b));
}
inline bool f16_lt(half a, half b) {
  ushort ai = as_type<ushort>(a), bi = as_type<ushort>(b);
  if (seismic_f16_nan(a) || seismic_f16_nan(b) ||
      (seismic_f16_zero(a) && seismic_f16_zero(b))) return false;
  bool as = (ai >> 15) != 0u, bs = (bi >> 15) != 0u;
  return as != bs ? as : (as ? ai > bi : ai < bi);
}
inline half f16_min(half a, half b) {
  if (seismic_f16_nan(a)) return b;
  if (seismic_f16_nan(b)) return a;
  if (seismic_f16_zero(a) && seismic_f16_zero(b))
    return as_type<half>(ushort(as_type<ushort>(a) | as_type<ushort>(b)));
  return f16_lt(b, a) ? b : a;
}
inline half f16_max(half a, half b) {
  if (seismic_f16_nan(a)) return b;
  if (seismic_f16_nan(b)) return a;
  if (seismic_f16_zero(a) && seismic_f16_zero(b))
    return as_type<half>(ushort(as_type<ushort>(a) & as_type<ushort>(b)));
  return f16_lt(a, b) ? b : a;
}
#if __METAL_VERSION__ >= 310
inline bool seismic_bf16_nan(bfloat value) {
  return (as_type<ushort>(value) & 0x7fffu) > 0x7f80u;
}
inline bool seismic_bf16_zero(bfloat value) {
  return (as_type<ushort>(value) & 0x7fffu) == 0u;
}
inline bool bf16_eq(bfloat a, bfloat b) {
  return !seismic_bf16_nan(a) && !seismic_bf16_nan(b) &&
         ((seismic_bf16_zero(a) && seismic_bf16_zero(b)) || as_type<ushort>(a) == as_type<ushort>(b));
}
inline bool bf16_lt(bfloat a, bfloat b) {
  ushort ai = as_type<ushort>(a), bi = as_type<ushort>(b);
  if (seismic_bf16_nan(a) || seismic_bf16_nan(b) ||
      (seismic_bf16_zero(a) && seismic_bf16_zero(b))) return false;
  bool as = (ai >> 15) != 0u, bs = (bi >> 15) != 0u;
  return as != bs ? as : (as ? ai > bi : ai < bi);
}
inline bfloat bf16_min(bfloat a, bfloat b) {
  if (seismic_bf16_nan(a)) return b;
  if (seismic_bf16_nan(b)) return a;
  if (seismic_bf16_zero(a) && seismic_bf16_zero(b))
    return as_type<bfloat>(ushort(as_type<ushort>(a) | as_type<ushort>(b)));
  return bf16_lt(b, a) ? b : a;
}
inline bfloat bf16_max(bfloat a, bfloat b) {
  if (seismic_bf16_nan(a)) return b;
  if (seismic_bf16_nan(b)) return a;
  if (seismic_bf16_zero(a) && seismic_bf16_zero(b))
    return as_type<bfloat>(ushort(as_type<ushort>(a) & as_type<ushort>(b)));
  return bf16_lt(a, b) ? b : a;
}
#endif

"#;
pub(crate) const SOFTFLOAT_PRELUDE: &str = include_str!("softfloat.metal");

/// Exact scalar instruction/helper family selected by the Metal renderer.
///
/// This is renderer-owned vocabulary: execution-service accounting can use it
/// without separately reconstructing which MSL form a closed operation emits.
/// `None` from one of the selectors below means that the operation belongs to
/// another explicitly typed family (for example strict F16/BF16 arithmetic),
/// not that rendering falls back heuristically.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScalarEmissionFamily {
    F32AddSub,
    F32Multiply,
    F32Divide,
    F32Remainder,
    F32MinMax,
    F32FusedMultiplyAdd,
    F32Comparison,
    F32ToF16,
    F16ToF32,
    F32ToBF16,
    BF16ToF32,
    F32ToInteger,
    IntegerToF32,
    NativeControl,
    NativeIntegerBit,
}

pub(crate) fn scalar_binary_emission_family(
    op: BinaryOp,
    ty: &ValueType,
) -> Option<ScalarEmissionFamily> {
    if matches!(ty, ValueType::Scalar(DType::F32)) {
        return Some(match op {
            BinaryOp::Add | BinaryOp::Sub => ScalarEmissionFamily::F32AddSub,
            BinaryOp::Mul => ScalarEmissionFamily::F32Multiply,
            BinaryOp::Div => ScalarEmissionFamily::F32Divide,
            BinaryOp::Rem => ScalarEmissionFamily::F32Remainder,
            BinaryOp::Min | BinaryOp::Max => ScalarEmissionFamily::F32MinMax,
        });
    }
    scalar_native_emission_family(ty)
}

pub(crate) fn scalar_fma_emission_family(ty: &ValueType) -> Option<ScalarEmissionFamily> {
    matches!(ty, ValueType::Scalar(DType::F32)).then_some(ScalarEmissionFamily::F32FusedMultiplyAdd)
}

pub(crate) fn scalar_comparison_emission_family(ty: &ValueType) -> Option<ScalarEmissionFamily> {
    if matches!(ty, ValueType::Scalar(DType::F32)) {
        Some(ScalarEmissionFamily::F32Comparison)
    } else {
        scalar_native_emission_family(ty)
    }
}

pub(crate) fn scalar_conversion_emission_family(
    from: &ValueType,
    to: &ValueType,
) -> Option<ScalarEmissionFamily> {
    let integer = |ty: &ValueType| {
        matches!(
            ty,
            ValueType::Scalar(DType::I32 | DType::U32) | ValueType::Index
        )
    };
    match (from, to) {
        (ValueType::Scalar(DType::F32), ValueType::Scalar(DType::F16)) => {
            Some(ScalarEmissionFamily::F32ToF16)
        }
        (ValueType::Scalar(DType::F16), ValueType::Scalar(DType::F32)) => {
            Some(ScalarEmissionFamily::F16ToF32)
        }
        (ValueType::Scalar(DType::F32), ValueType::Scalar(DType::BF16)) => {
            Some(ScalarEmissionFamily::F32ToBF16)
        }
        (ValueType::Scalar(DType::BF16), ValueType::Scalar(DType::F32)) => {
            Some(ScalarEmissionFamily::BF16ToF32)
        }
        (ValueType::Scalar(DType::F32), destination) if integer(destination) => {
            Some(ScalarEmissionFamily::F32ToInteger)
        }
        (source, ValueType::Scalar(DType::F32)) if integer(source) => {
            Some(ScalarEmissionFamily::IntegerToF32)
        }
        _ if matches!(from, ValueType::Bool | ValueType::Scalar(DType::Bool))
            || matches!(to, ValueType::Bool | ValueType::Scalar(DType::Bool)) =>
        {
            Some(ScalarEmissionFamily::NativeControl)
        }
        _ if integer(from) && integer(to) => Some(ScalarEmissionFamily::NativeIntegerBit),
        _ => None,
    }
}

pub(crate) const fn scalar_control_emission_family() -> ScalarEmissionFamily {
    ScalarEmissionFamily::NativeControl
}

pub(crate) fn scalar_integer_bit_emission_family(ty: &ValueType) -> Option<ScalarEmissionFamily> {
    matches!(
        ty,
        ValueType::Scalar(DType::I32 | DType::U32) | ValueType::Index
    )
    .then_some(ScalarEmissionFamily::NativeIntegerBit)
}

fn scalar_native_emission_family(ty: &ValueType) -> Option<ScalarEmissionFamily> {
    if matches!(ty, ValueType::Bool | ValueType::Scalar(DType::Bool)) {
        Some(ScalarEmissionFamily::NativeControl)
    } else {
        scalar_integer_bit_emission_family(ty)
    }
}

#[derive(Debug)]
pub(crate) struct RenderedKernel {
    pub name: String,
    pub source: String,
}

/// Renders one kernel. `index` names the function.
pub(crate) fn render(
    index: usize,
    kernel: &Kernel<Metal>,
    shape: &KernelEmissionLayout,
) -> RenderedKernel {
    let mut renderer = Renderer::new(kernel, shape);
    let name = format!("seismic_k{index}");
    let source = renderer.render_function(&name);
    RenderedKernel { name, source }
}

// ---------------------------------------------------------------------------
// Storage descriptions the renderer derives once
// ---------------------------------------------------------------------------

struct GlobalPlace {
    pointer: String,
    geometry: seismic_compiler::target::RepresentationGeometry,
    rank: u32,
    /// Byte pointer name (`seismic_b{slot}`) for word-granular atomics.
    base: String,
    /// Word index of the offset word in the parameter block.
    word: usize,
}

enum LocalStorage {
    Pointer { pointer: String },
}

struct LocalPlace {
    storage: LocalStorage,
    dtype: DType,
    geometry: seismic_compiler::target::RepresentationGeometry,
    strides: Vec<String>,
}

trait MetalGeometry {
    fn info(&self) -> &'static seismic_lang::registry::RepresentationInfo;
    fn decode(&self) -> Option<&seismic_lang::registry::DecodeRecipe> {
        None
    }
}

impl MetalGeometry for seismic_compiler::target::RepresentationGeometry {
    fn info(&self) -> &'static seismic_lang::registry::RepresentationInfo {
        self.info
    }
    fn decode(&self) -> Option<&seismic_lang::registry::DecodeRecipe> {
        self.decode.as_ref()
    }
}
impl MetalGeometry for seismic_compiler::target::DenseRepresentationGeometry {
    fn info(&self) -> &'static seismic_lang::registry::RepresentationInfo {
        self.info
    }
}
impl MetalGeometry for seismic_compiler::target::PackedRepresentationGeometry {
    fn info(&self) -> &'static seismic_lang::registry::RepresentationInfo {
        self.info
    }
    fn decode(&self) -> Option<&seismic_lang::registry::DecodeRecipe> {
        Some(&self.decode)
    }
}
impl MetalGeometry for seismic_compiler::target::ReadableRepresentationGeometry {
    fn info(&self) -> &'static seismic_lang::registry::RepresentationInfo {
        match self {
            Self::Dense(geometry) => geometry.info,
            Self::Packed(geometry) => geometry.info,
        }
    }
    fn decode(&self) -> Option<&seismic_lang::registry::DecodeRecipe> {
        match self {
            Self::Dense(_) => None,
            Self::Packed(geometry) => Some(&geometry.decode),
        }
    }
}

struct Renderer<'a> {
    kernel: &'a Kernel<Metal>,
    shape: &'a KernelEmissionLayout,
    names: BTreeMap<ErasedValue, String>,
    globals: Vec<GlobalPlace>,
    locals: Vec<LocalPlace>,
    temporaries: u32,
}

impl<'a> Renderer<'a> {
    fn new(kernel: &'a Kernel<Metal>, shape: &'a KernelEmissionLayout) -> Self {
        let interface = kernel.interface();
        let mut globals = Vec::new();
        for (position, binding) in interface.bindings.iter().enumerate() {
            let geometry = shape.bindings[position].geometry.clone();
            let rank = binding.rank;
            globals.push(GlobalPlace {
                pointer: format!("seismic_p{}", binding.slot.ordinal()),
                geometry,
                rank,
                base: format!("seismic_b{}", binding.slot.ordinal()),
                word: shape.words.bindings[position].first as usize,
            });
        }
        let mut locals = Vec::new();
        for (index, local) in kernel.locals().iter().enumerate() {
            let geometry = shape.locals[index].geometry.clone();
            let dtype = geometry.info.decoded;
            let strides: Vec<String> = (0..local.extents.len())
                .map(|axis| format!("seismic_ls{index}_{axis}"))
                .collect();
            let storage = match local.kind {
                LaunchLocalKind::Workgroup => LocalStorage::Pointer {
                    pointer: format!("seismic_wg{index}"),
                },
                LaunchLocalKind::Participant => LocalStorage::Pointer {
                    pointer: format!("seismic_part{index}"),
                },
                LaunchLocalKind::Register => LocalStorage::Pointer {
                    pointer: format!("seismic_reg{index}"),
                },
            };
            locals.push(LocalPlace {
                storage,
                dtype,
                geometry,
                strides,
            });
        }
        Self {
            kernel,
            shape,
            names: BTreeMap::new(),
            globals,
            locals,
            temporaries: 0,
        }
    }

    // -- the function ---------------------------------------------------------

    fn render_function(&mut self, name: &str) -> String {
        let interface = self.kernel.interface();
        let mut parameters = Vec::new();
        for binding in &interface.bindings {
            let address = match binding.access {
                seismic_compiler::kernel::ops::BindingAccess::Read => "const device",
                seismic_compiler::kernel::ops::BindingAccess::Write => "device",
            };
            parameters.push(format!(
                "{address} char* seismic_b{slot} [[buffer({slot})]]",
                slot = binding.slot.ordinal()
            ));
        }
        parameters.push(format!(
            "const device ulong* seismic_params [[buffer({})]]",
            interface.bindings.len()
        ));
        parameters.push(format!(
            "device ulong* seismic_results [[buffer({})]]",
            interface.bindings.len() + 1
        ));
        parameters.push(format!(
            "device char* seismic_participant [[buffer({})]]",
            interface.bindings.len() + 2
        ));
        parameters.push(format!(
            "device char* seismic_register [[buffer({})]]",
            interface.bindings.len() + 3
        ));
        parameters.push("threadgroup char* seismic_workgroup [[threadgroup(0)]]".into());
        parameters.push("uint3 seismic_tg [[threadgroup_position_in_grid]]".into());
        parameters.push("uint3 seismic_tid [[thread_position_in_threadgroup]]".into());
        parameters.push("uint3 seismic_tpg [[threads_per_threadgroup]]".into());
        parameters.push("uint3 seismic_tgn [[threadgroups_per_grid]]".into());
        parameters.push("uint seismic_lane [[thread_index_in_simdgroup]]".into());
        parameters.push("uint seismic_simd_width [[threads_per_simdgroup]]".into());

        let mut body: Vec<String> = Vec::new();
        // Typed pointers and layout words of every binding.
        for (slot, place) in self.globals.iter().enumerate() {
            let access = match interface.bindings[slot].access {
                seismic_compiler::kernel::ops::BindingAccess::Read => "const device",
                seismic_compiler::kernel::ops::BindingAccess::Write => "device",
            };
            let pointer_type = match place.geometry.info.kind {
                RepresentationKind::Dense(dtype) => dtype_name(dtype),
                RepresentationKind::Packed(_) => "uchar",
                RepresentationKind::External(_) => "uchar",
            };
            body.push(format!(
                "{access} {ty}* {p} = reinterpret_cast<{access} {ty}*>({b});",
                ty = pointer_type,
                p = place.pointer,
                b = place.base
            ));
            for axis in 0..place.rank as usize {
                body.push(format!(
                    "const ulong seismic_e{slot}_{axis} = seismic_params[{}];",
                    place.word + axis
                ));
                body.push(format!(
                    "const ulong seismic_s{slot}_{axis} = seismic_params[{}];",
                    place.word + place.rank as usize + axis
                ));
            }
        }
        body.push("const ulong seismic_group_linear = (ulong(seismic_tg.z) * ulong(seismic_tgn.y) + ulong(seismic_tg.y)) * ulong(seismic_tgn.x) + ulong(seismic_tg.x);".into());
        body.push("const ulong seismic_thread_linear = (ulong(seismic_tid.z) * ulong(seismic_tpg.y) + ulong(seismic_tid.y)) * ulong(seismic_tpg.x) + ulong(seismic_tid.x);".into());
        body.push("const ulong seismic_threads_per_group = ulong(seismic_tpg.x) * ulong(seismic_tpg.y) * ulong(seismic_tpg.z);".into());
        body.push("const ulong seismic_participant_linear = seismic_group_linear * seismic_threads_per_group + seismic_thread_linear;".into());
        let locals = self.kernel.locals();
        for (index, local) in locals.iter().enumerate() {
            let words = self.shape.words.locals[index];
            body.push(format!(
                "const ulong seismic_lo{index} = seismic_params[{}];",
                words.first
            ));
            for axis in 0..local.extents.len() {
                body.push(format!(
                    "const ulong seismic_le{index}_{axis} = seismic_params[{}];",
                    words.first + 1 + axis as u32
                ));
                body.push(format!(
                    "const ulong seismic_ls{index}_{axis} = seismic_params[{}];",
                    words.first + 1 + words.rank + axis as u32
                ));
            }
            let (address, space) = match local.kind {
                LaunchLocalKind::Workgroup => (format!("seismic_workgroup + seismic_lo{index}"), "threadgroup"),
                LaunchLocalKind::Participant => (format!("seismic_participant + seismic_participant_linear * seismic_params[{}] + seismic_lo{index}", self.shape.words.local_total_first + 1), "device"),
                LaunchLocalKind::Register => (format!("seismic_register + seismic_participant_linear * seismic_params[{}] + seismic_lo{index}", self.shape.words.local_total_first + 2), "device"),
            };
            let LocalStorage::Pointer { pointer } = &self.locals[index].storage;
            body.push(format!(
                "{space} {}* {pointer} = reinterpret_cast<{space} {}*>({address});",
                dtype_name(self.locals[index].dtype),
                dtype_name(self.locals[index].dtype),
            ));
        }
        let root = self.kernel.root();
        self.render_block(root, &[], &mut body);

        let mut source = String::new();
        let _ = writeln!(
            source,
            "kernel void {name}(\n    {}\n) {{",
            parameters.join(",\n    ")
        );
        for line in body {
            let _ = writeln!(source, "    {line}");
        }
        source.push_str("}\n\n");
        source
    }

    // -- names and types --------------------------------------------------------

    fn name(&self, value: ErasedValue) -> &str {
        self.names
            .get(&value)
            .map(String::as_str)
            .unwrap_or_else(|| panic!("kernel value {value:?} is used before its defining op in the same or a dominating block"))
    }

    fn define(&mut self, value: ErasedValue) -> String {
        let name = format!("v{}", self.names.len());
        self.names.insert(value, name.clone());
        name
    }

    fn value_type(&self, value: ErasedValue) -> ValueType {
        self.kernel.value_type(value)
    }

    fn type_name(&self, value: ErasedValue) -> &'static str {
        value_type_name(&self.value_type(value))
    }

    fn temporary(&mut self) -> String {
        let temp = format!("seismic_t{}", self.temporaries);
        self.temporaries += 1;
        temp
    }

    /// Declares `out` with `expression`.
    fn assign(&mut self, out: ErasedValue, expression: String, lines: &mut Vec<String>) {
        let ty = self.type_name(out);
        let name = self.define(out);
        lines.push(format!("{ty} {name} = {expression};"));
    }

    // -- addresses ----------------------------------------------------------------

    /// `(pointer, element index expression)` of a place at coordinates.
    fn address(&self, place: PlaceRef, coords: &[String]) -> (String, String) {
        match place {
            PlaceRef::Global { slot } => {
                let global = &self.globals[slot.ordinal() as usize];
                let terms: Vec<String> = coords
                    .iter()
                    .enumerate()
                    .map(|(axis, coord)| {
                        let coordinate = if axis + 1 == coords.len() {
                            match &global.geometry.info.kind {
                                RepresentationKind::Packed(layout) => {
                                    format!("(ulong({coord}) / {}ul)", layout.group)
                                }
                                RepresentationKind::Dense(_) => format!("ulong({coord})"),
                                RepresentationKind::External(layout) => {
                                    format!("(ulong({coord}) / {}ul)", layout.logical_group)
                                }
                            }
                        } else {
                            format!("ulong({coord})")
                        };
                        format!("{coordinate} * seismic_s{}_{axis}", slot.ordinal())
                    })
                    .collect();
                let units = sum(&terms);
                let address = match &global.geometry.info.kind {
                    RepresentationKind::Dense(_) => units,
                    RepresentationKind::Packed(layout) => {
                        format!("({units}) * {}ul", layout.packet_size)
                    }
                    RepresentationKind::External(layout) => {
                        format!("({units}) * {}ul", layout.packet_size)
                    }
                };
                (global.pointer.clone(), address)
            }
            PlaceRef::Local { index } => {
                let local = &self.locals[index as usize];
                let terms = coords
                    .iter()
                    .enumerate()
                    .map(|(axis, coord)| format!("ulong({coord}) * {}", local.strides[axis]))
                    .collect::<Vec<_>>();
                let LocalStorage::Pointer { pointer } = &local.storage;
                (pointer.clone(), sum(&terms))
            }
        }
    }

    fn closed_address<G: MetalGeometry>(
        &self,
        place: &ClosedPlace<G>,
        coords: &[String],
    ) -> (String, String) {
        match place.kind {
            ClosedPlaceKind::Global { buffer_ordinal, .. } => {
                let global = &self.globals[buffer_ordinal as usize];
                let terms = coords
                    .iter()
                    .enumerate()
                    .map(|(axis, coord)| {
                        let coordinate = if axis + 1 == coords.len() {
                            match &place.geometry.info().kind {
                                RepresentationKind::Packed(layout) => {
                                    format!("(ulong({coord}) / {}ul)", layout.group)
                                }
                                RepresentationKind::External(layout) => {
                                    format!("(ulong({coord}) / {}ul)", layout.logical_group)
                                }
                                RepresentationKind::Dense(_) => format!("ulong({coord})"),
                            }
                        } else {
                            format!("ulong({coord})")
                        };
                        format!("{coordinate} * seismic_s{buffer_ordinal}_{axis}")
                    })
                    .collect::<Vec<_>>();
                let units = sum(&terms);
                let address = match &place.geometry.info().kind {
                    RepresentationKind::Dense(_) => units,
                    RepresentationKind::Packed(layout) => {
                        format!("({units}) * {}ul", layout.packet_size)
                    }
                    RepresentationKind::External(layout) => {
                        format!("({units}) * {}ul", layout.packet_size)
                    }
                };
                (global.pointer.clone(), address)
            }
            ClosedPlaceKind::Local { index, .. } => {
                let local = &self.locals[index as usize];
                let terms = coords
                    .iter()
                    .zip(&local.strides)
                    .map(|(coord, stride)| format!("ulong({coord}) * {stride}"))
                    .collect::<Vec<_>>();
                let LocalStorage::Pointer { pointer } = &local.storage;
                (pointer.clone(), sum(&terms))
            }
        }
    }

    fn coords(&self, index: &[ErasedValue]) -> Vec<String> {
        index
            .iter()
            .map(|value| self.name(*value).to_string())
            .collect()
    }

    fn logical_address(&self, tensor: &LogicalTensorMap, coords: &[String]) -> (String, String) {
        let coords = self.logical_coords(tensor, coords);
        self.address(tensor.base, &coords)
    }

    fn logical_read_expression(&self, tensor: &LogicalTensorMap, coords: &[String]) -> String {
        let coords = self.logical_coords(tensor, coords);
        let place = self.kernel.closed_place(tensor.base, self.shape);
        self.read_expression(&place, &coords)
    }

    fn logical_coords(&self, tensor: &LogicalTensorMap, coords: &[String]) -> Vec<String> {
        let mut coords = coords.to_vec();
        for step in tensor.steps.iter().rev() {
            coords = match step {
                LogicalViewStep::Slice(axes) => {
                    let mut logical = coords.iter();
                    axes.iter()
                        .map(|axis| match axis {
                            LogicalSliceAxis::Point(value) => self.name(*value).to_string(),
                            LogicalSliceAxis::Range { start, .. } => format!(
                                "({} + {})",
                                self.name(*start),
                                logical.next().expect("closed slice-map rank")
                            ),
                            LogicalSliceAxis::Full => {
                                logical.next().expect("closed slice-map rank").clone()
                            }
                        })
                        .collect()
                }
                LogicalViewStep::Transpose(permutation) => {
                    let mut physical = vec![String::new(); permutation.len()];
                    for (source_axis, coordinate) in permutation.iter().zip(&coords) {
                        physical[*source_axis as usize] = coordinate.clone();
                    }
                    physical
                }
                LogicalViewStep::Reshape { from, to } => {
                    let mut linear = "0ul".to_string();
                    for (coordinate, extent) in coords.iter().zip(to) {
                        linear = format!(
                            "(({linear}) * ulong({}) + ulong({coordinate}))",
                            self.name(*extent)
                        );
                    }
                    let mut physical = vec![String::new(); from.len()];
                    for axis in (0..from.len()).rev() {
                        physical[axis] = format!("(({linear}) % ulong({}))", self.name(from[axis]));
                        linear = format!("(({linear}) / ulong({}))", self.name(from[axis]));
                    }
                    physical
                }
            };
        }
        coords
    }

    fn read_expression<G: MetalGeometry>(
        &self,
        place: &ClosedPlace<G>,
        coords: &[String],
    ) -> String {
        let (pointer, address) = self.closed_address(place, coords);
        let geometry = &place.geometry;
        match &geometry.info().kind {
            RepresentationKind::Dense(_) => format!("{pointer}[{address}]"),
            RepresentationKind::Packed(layout) => {
                let recipe = geometry
                    .decode()
                    .expect("packed geometry has a registry decode recipe");
                let mut temps: Vec<String> = Vec::with_capacity(recipe.temporary_count());
                for step in recipe.steps() {
                    let expression = match step {
                        DecodeStep::ReadPlaneField { plane, field, .. } => self
                            .read_plane_expression(place, layout, *plane as usize, coords, *field),
                        DecodeStep::InterpretCode {
                            raw,
                            bits,
                            interpretation,
                            ..
                        } => interpret_code(&temps[recipe.ordinal(*raw)], *bits, interpretation),
                        DecodeStep::DecodeFloatCode { raw, format, .. } => format!(
                            "{}({})",
                            float_code_function(*format),
                            temps[recipe.ordinal(*raw)]
                        ),
                        DecodeStep::ConvertToF32 { from, .. } => match recipe.dtype(*from) {
                            DType::F32 => temps[recipe.ordinal(*from)].clone(),
                            DType::F16 => {
                                format!("f16_to_f32({})", temps[recipe.ordinal(*from)])
                            }
                            DType::BF16 => {
                                format!("seismic_bf16_widen({})", temps[recipe.ordinal(*from)])
                            }
                            _ => format!("float({})", temps[recipe.ordinal(*from)]),
                        },
                        DecodeStep::Multiply { left, right, .. } => format!(
                            "f32_mul(float({}), float({}))",
                            temps[recipe.ordinal(*left)],
                            temps[recipe.ordinal(*right)]
                        ),
                        DecodeStep::Negate { from, .. } => {
                            format!(
                                "as_type<float>(as_type<uint>(float({})) ^ 0x80000000u)",
                                temps[recipe.ordinal(*from)]
                            )
                        }
                        DecodeStep::MultiplyAdd {
                            factor,
                            multiplicand,
                            addend,
                            ..
                        } => format!(
                            "f32_mulAdd(float({}), float({}), float({}))",
                            temps[recipe.ordinal(*factor)],
                            temps[recipe.ordinal(*multiplicand)],
                            temps[recipe.ordinal(*addend)]
                        ),
                        DecodeStep::Cast { from, to, .. } => match to {
                            DType::F32 => temps[recipe.ordinal(*from)].clone(),
                            DType::F16 => {
                                format!("f32_to_f16({})", temps[recipe.ordinal(*from)])
                            }
                            DType::BF16 => {
                                format!("seismic_bf16_narrow({})", temps[recipe.ordinal(*from)])
                            }
                            _ => format!("{}({})", dtype_name(*to), temps[recipe.ordinal(*from)]),
                        },
                    };
                    temps.push(format!("({expression})"));
                }
                temps[recipe.ordinal(recipe.output())].clone()
            }
            RepresentationKind::External(_) => {
                panic!("external packets are consumed only by registered conversion ops")
            }
        }
    }

    fn read_plane_expression<G: MetalGeometry>(
        &self,
        place: &ClosedPlace<G>,
        layout: &seismic_lang::registry::PackedPacketLayout,
        plane: usize,
        coords: &[String],
        field: u32,
    ) -> String {
        let (pointer, packet) = self.closed_address(place, coords);
        let schema = &layout.planes[plane];
        let logical = coords.last().map(String::as_str).unwrap_or("0");
        let entry = format!(
            "((ulong({logical}) % {}ul) / {}ul * {}ul + {}ul)",
            layout.group, schema.group, schema.fields, field
        );
        let base = format!("({pointer} + ({packet}) + {}ul)", schema.offset);
        match &schema.encoding {
            PlaneEncoding::Dense(dtype) => format!(
                "(*reinterpret_cast<const device {}*>({base} + ({entry}) * {}ul))",
                dtype_name(*dtype),
                dtype.bytes()
            ),
            PlaneEncoding::Packed { bits, .. } => {
                let bit = format!("(({entry}) * {bits}ul)");
                let mask = if *bits == 32 {
                    "0xffffffffu".to_string()
                } else {
                    format!("{}u", (1u32 << bits) - 1)
                };
                format!("((reinterpret_cast<const device uint*>({base})[({bit}) / 32ul] >> uint(({bit}) % 32ul)) & {mask})")
            }
            PlaneEncoding::FloatCode { format } => {
                let bits = format.bits();
                let bit = format!("(({entry}) * {bits}ul)");
                let mask = format!("{}u", (1u32 << bits) - 1);
                format!("((uint({base}[({bit}) / 8ul]) >> uint(({bit}) % 8ul)) & {mask})")
            }
        }
    }

    fn read_plane_raw_expression(
        &self,
        place: &seismic_compiler::kernel::ops::ClosedPackedPlace,
        plane: usize,
        coords: &[String],
    ) -> String {
        let geometry = &place.geometry;
        let layout = &geometry.layout;
        let schema = &layout.planes[plane];
        let decoded = self.read_plane_expression(place, layout, plane, coords, 0);
        match &schema.encoding {
            PlaneEncoding::Dense(DType::F32 | DType::I32 | DType::U32) => {
                format!("as_type<uint>({decoded})")
            }
            PlaneEncoding::Dense(DType::F16 | DType::BF16) => {
                format!("uint(as_type<ushort>({decoded}))")
            }
            PlaneEncoding::Dense(DType::Bool) => format!("uint({decoded})"),
            PlaneEncoding::Packed { .. } | PlaneEncoding::FloatCode { .. } => decoded,
        }
    }

    // -- blocks ---------------------------------------------------------------------

    /// Renders a block; `Yield` assigns to `targets`.
    fn render_block(&mut self, block: BlockId, targets: &[String], lines: &mut Vec<String>) {
        let block: &Block<Metal> = self.kernel.block(block);
        for op in &block.ops {
            self.render_op(self.kernel.closed_op(op, self.shape), targets, lines);
        }
    }

    fn render_op(
        &mut self,
        op: ClosedOpView<'_, Metal>,
        targets: &[String],
        lines: &mut Vec<String>,
    ) {
        let raw = |value: ClosedValue| value.value;
        match op {
            ClosedOpView::Constant { out, value } => {
                let expression = constant_expr(value, &out.ty);
                self.assign(raw(out), expression, lines);
            }
            ClosedOpView::Binary { op, out, a, b } => {
                let a_name = self.name(raw(a)).to_owned();
                let b_name = self.name(raw(b)).to_owned();
                let expression =
                    binary_expr(op, &a.ty, &a_name, &b_name, &mut || self.temporary_name());
                self.assign(raw(out), expression, lines);
            }
            ClosedOpView::Unary { op, out, a } => {
                let name = self.name(raw(a)).to_string();
                let expression = match (op, scalar_kind(&a.ty)) {
                    (UnaryOp::Neg, Kind::Float) => float_sign_op(&a.ty, &name, true),
                    (UnaryOp::Neg, Kind::Int | Kind::Bool) => {
                        format!(
                            "as_type<{}>(0u - as_type<uint>({name}))",
                            value_type_name(&a.ty)
                        )
                    }
                    (UnaryOp::Abs, Kind::Float) => float_sign_op(&a.ty, &name, false),
                    (UnaryOp::Abs, Kind::Int | Kind::Bool) => format!("abs({name})"),
                };
                self.assign(raw(out), expression, lines);
            }
            ClosedOpView::Bit { op, out, a, b } => {
                let ty = a.ty;
                let name = value_type_name(&ty);
                let (a, b) = (self.name(raw(a)).to_string(), self.name(raw(b)).to_string());
                let expression = match op {
                    BitOp::And => {
                        format!("as_type<{name}>(as_type<uint>({a}) & as_type<uint>({b}))")
                    }
                    BitOp::Or => {
                        format!("as_type<{name}>(as_type<uint>({a}) | as_type<uint>({b}))")
                    }
                    BitOp::Xor => {
                        format!("as_type<{name}>(as_type<uint>({a}) ^ as_type<uint>({b}))")
                    }
                    BitOp::Shl => format!("as_type<{name}>(as_type<uint>({a}) << uint({b}))"),
                    BitOp::Shr => {
                        if matches!(ty, ValueType::Scalar(DType::I32)) {
                            format!("as_type<{name}>(as_type<int>({a}) >> uint({b}))")
                        } else {
                            format!("as_type<{name}>(as_type<uint>({a}) >> uint({b}))")
                        }
                    }
                };
                self.assign(raw(out), expression, lines);
            }
            ClosedOpView::Fma { out, a, b, c } => {
                let function = fma_function(&a.ty);
                let expression = format!(
                    "{function}({}, {}, {})",
                    self.name(raw(a)),
                    self.name(raw(b)),
                    self.name(raw(c))
                );
                self.assign(raw(out), expression, lines);
            }
            ClosedOpView::VectorSplat { .. }
            | ClosedOpView::VectorBinary { .. }
            | ClosedOpView::VectorUnary { .. }
            | ClosedOpView::VectorBit { .. }
            | ClosedOpView::VectorFma { .. }
            | ClosedOpView::VectorCast { .. }
            | ClosedOpView::VectorLane { .. }
            | ClosedOpView::VectorReduceAdd { .. }
            | ClosedOpView::VectorRead { .. }
            | ClosedOpView::VectorWrite { .. } => {
                panic!("closed Metal kernel contains a vector operation although the target profile advertises an empty VectorSupport")
            }
            ClosedOpView::ApproximateMath { op, out, a } => {
                let expression = math_expr(op, &a.ty, self.name(raw(a)));
                self.assign(raw(out), expression, lines);
            }
            ClosedOpView::Cast { out, a, to } => {
                let expression = cast_expr(&a.ty, &to, self.name(raw(a)));
                self.assign(raw(out), expression, lines);
            }
            ClosedOpView::Bitcast { out, a, to } => {
                self.assign(
                    raw(out),
                    format!("as_type<{}>({})", value_type_name(&to), self.name(raw(a))),
                    lines,
                );
            }
            ClosedOpView::Cmp { op, out, a, b } => {
                let expression = compare_expr(op, &a.ty, self.name(raw(a)), self.name(raw(b)));
                self.assign(out.value(), expression, lines);
            }
            ClosedOpView::Select {
                out,
                condition,
                a,
                b,
            } => {
                let expression = format!(
                    "(({}) ? ({}) : ({}))",
                    self.name(condition.value()),
                    self.name(raw(a)),
                    self.name(raw(b))
                );
                self.assign(raw(out), expression, lines);
            }
            ClosedOpView::Logic { op, out, a, b } => {
                let symbol = match op {
                    LogicOp::And => "&&",
                    LogicOp::Or => "||",
                };
                let expression = format!(
                    "({} {symbol} {})",
                    self.name(a.value()),
                    self.name(b.value())
                );
                self.assign(out.value(), expression, lines);
            }
            ClosedOpView::Not { out, a } => {
                let expression = format!("!({})", self.name(a.value()));
                self.assign(out.value(), expression, lines);
            }
            ClosedOpView::Geometry { out, kind } => {
                let expression = geometry_expr(kind);
                self.assign(out.value(), expression, lines);
            }
            ClosedOpView::NatArg { out, index, .. } => {
                let expression = format!(
                    "uint(seismic_params[{}])",
                    self.shape.words.nat_first + index
                );
                self.assign(out.value(), expression, lines);
            }
            ClosedOpView::ScalarArg {
                out, index, dtype, ..
            } => {
                let word = format!("seismic_params[{}]", self.shape.words.scalar_first + index);
                let expression = decode_word(dtype, &word);
                self.assign(raw(out), expression, lines);
            }
            ClosedOpView::Read {
                out,
                place,
                indices,
            } => {
                let raw_indices = indices
                    .iter()
                    .map(|value| value.value())
                    .collect::<Vec<_>>();
                let coords = self.coords(&raw_indices);
                let expression = self.read_expression(&place, &coords);
                self.assign(raw(out), expression, lines);
            }
            ClosedOpView::ReadPlane {
                out,
                place,
                plane,
                indices,
                ..
            } => {
                let raw_indices = indices
                    .iter()
                    .map(|value| value.value())
                    .collect::<Vec<_>>();
                let coords = self.coords(&raw_indices);
                let expression = self.read_plane_raw_expression(&place, plane as usize, &coords);
                self.assign(raw(out), expression, lines);
            }
            ClosedOpView::RepresentationConvertPacket {
                source,
                destination,
                recipe,
                packet,
                ..
            } => {
                self.render_representation_conversion(
                    &source,
                    &destination,
                    recipe,
                    packet.value(),
                    lines,
                );
            }
            ClosedOpView::Write {
                place,
                indices,
                value,
            } => {
                let raw_indices = indices
                    .iter()
                    .map(|value| value.value())
                    .collect::<Vec<_>>();
                let coords = self.coords(&raw_indices);
                let (pointer, address) = self.closed_address(&place, &coords);
                let dtype = place.geometry.dtype;
                lines.push(format!(
                    "{pointer}[{address}] = {}({});",
                    dtype_name(dtype),
                    self.name(raw(value))
                ));
            }
            ClosedOpView::Extent { out, place, axis } => {
                let expression = match place.kind {
                    ClosedPlaceKind::Global { buffer_ordinal, .. } => {
                        format!("uint(seismic_e{buffer_ordinal}_{axis})")
                    }
                    ClosedPlaceKind::Local { index, .. } => {
                        format!("uint(seismic_le{index}_{axis})")
                    }
                };
                self.assign(out.value(), expression, lines);
            }
            ClosedOpView::Atomic {
                op,
                place,
                indices,
                value,
            } => {
                let raw_indices = indices
                    .iter()
                    .map(|value| value.value())
                    .collect::<Vec<_>>();
                self.render_atomic(op, &place, &raw_indices, raw(value), lines);
            }
            ClosedOpView::StoreSlot {
                slot,
                dtype,
                value,
                election: seismic_compiler::kernel::ops::StoreElection::GlobalLeader,
            } => {
                let bits = encode_word(dtype, self.name(raw(value)));
                lines.push(format!("if (seismic_participant_linear == 0ul) seismic_results[{slot}] = ulong({bits});"));
            }
            ClosedOpView::Barrier(scope) => lines.push(match scope {
                BarrierScope::Workgroup => {
                    "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);"
                        .into()
                }
                BarrierScope::Subgroup => {
                    "simdgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);".into()
                }
            }),
            ClosedOpView::Intrinsic {
                op,
                outputs,
                arguments,
                ..
            } => {
                let outputs = outputs.iter().map(|value| value.value).collect::<Vec<_>>();
                let arguments = arguments
                    .iter()
                    .map(|value| value.value)
                    .collect::<Vec<_>>();
                self.render_intrinsic(op, &outputs, &arguments, lines)
            }
            ClosedOpView::Branch {
                condition,
                then_block,
                else_block,
                outputs,
                ..
            } => {
                let mut names = Vec::with_capacity(outputs.len());
                for out in outputs {
                    let name = self.define(raw(out));
                    lines.push(format!(
                        "{} {name} = {};",
                        value_type_name(&out.ty),
                        zero_of(&out.ty)
                    ));
                    names.push(name);
                }
                lines.push(format!("if ({}) {{", self.name(condition.value())));
                self.render_block(then_block, &names, lines);
                lines.push("} else {".into());
                self.render_block(else_block, &names, lines);
                lines.push("}".into());
            }
            ClosedOpView::Repeat {
                start,
                end,
                binder,
                carries_in,
                carry_parameters,
                body,
                outputs,
                ..
            } => {
                let mut names = Vec::with_capacity(outputs.len());
                for (out, initial) in outputs.into_iter().zip(carries_in) {
                    let initial = self.name(raw(initial)).to_string();
                    let name = self.define(raw(out));
                    lines.push(format!("{} {name} = {initial};", value_type_name(&out.ty)));
                    names.push(name);
                }
                let binder_name = self.define(binder.value());
                lines.push(format!(
                    "for (uint {binder_name} = {}; {binder_name} < {}; {binder_name}++) {{",
                    self.name(start.value()),
                    self.name(end.value())
                ));
                for (parameter, name) in carry_parameters.iter().zip(&names) {
                    self.names.insert(parameter.value, name.clone());
                }
                self.render_block(body, &names, lines);
                lines.push("}".into());
            }
            ClosedOpView::Yield { values } => {
                for (target, value) in targets.iter().zip(values) {
                    lines.push(format!("{target} = {};", self.name(raw(value))));
                }
            }
        }
    }

    fn temporary_name(&mut self) -> String {
        self.temporary()
    }

    fn render_representation_conversion(
        &mut self,
        source: &seismic_compiler::kernel::ops::ClosedExternalGlobalPlace,
        destination: &seismic_compiler::kernel::ops::ClosedPackedGlobalPlace,
        conversion: &seismic_lang::registry::RepresentationConversion,
        packet: ErasedValue,
        lines: &mut Vec<String>,
    ) {
        let source_layout = &source.geometry.layout;
        let destination_layout = &destination.geometry.layout;
        let source_pointer = self.globals[source.buffer_ordinal as usize].pointer.clone();
        let destination_pointer = self.globals[destination.buffer_ordinal as usize]
            .pointer
            .clone();
        let packet = self.name(packet).to_string();
        let source_base = format!(
            "({source_pointer} + ulong({packet}) * {}ul)",
            source_layout.packet_size
        );
        let destination_base = format!(
            "({destination_pointer} + ulong({packet}) * {}ul)",
            destination_layout.packet_size
        );
        for (plane_index, plane_recipe) in conversion.recipe.planes.iter().enumerate() {
            let plane = &destination_layout.planes[plane_index];
            match plane_recipe {
                PlaneRepackRecipe::BitRoutes(routes) => {
                    for (byte, bits) in routes.chunks(8).enumerate() {
                        let terms = bits
                            .iter()
                            .enumerate()
                            .map(|(destination_bit, source_bit)| {
                                format!(
                                    "((uint({source_base}[{}ul]) >> {}u) & 1u) << {}u",
                                    source_bit / 8,
                                    source_bit % 8,
                                    destination_bit
                                )
                            })
                            .collect::<Vec<_>>();
                        lines.push(format!(
                            "{destination_base}[{}ul] = uchar({});",
                            u64::from(plane.offset) + byte as u64,
                            if terms.is_empty() {
                                "0u".into()
                            } else {
                                terms.join(" | ")
                            }
                        ));
                    }
                }
                PlaneRepackRecipe::DenseValues(values) => {
                    let PlaneEncoding::Dense(dtype) = &plane.encoding else {
                        panic!("dense conversion recipe targets a non-dense plane")
                    };
                    for (element, expression) in values.iter().enumerate() {
                        let expression = repack_expression(expression, &source_base);
                        let expression = strict_f32_publication(*dtype, &expression);
                        let offset =
                            u64::from(plane.offset) + element as u64 * u64::from(dtype.bytes());
                        lines.push(format!(
                            "*reinterpret_cast<device {}*>({destination_base} + {offset}ul) = {expression};",
                            dtype_name(*dtype),
                        ));
                    }
                }
            }
        }
    }

    // -- atomics ----------------------------------------------------------------------

    /// `Add` rounds once per update at the element dtype; `Max`/`Min` are
    /// exact. 32-bit integers use the native fetch ops; floating elements
    /// use a compare-exchange loop on the containing 32-bit word.
    fn render_atomic(
        &mut self,
        op: AtomicOp,
        place: &seismic_compiler::kernel::ops::ClosedDensePlace,
        index: &[ErasedValue],
        value: ErasedValue,
        lines: &mut Vec<String>,
    ) {
        let coords = self.coords(index);
        let (pointer, address) = self.closed_address(place, &coords);
        let dtype = place.geometry.dtype;
        let value = self.name(value).to_string();
        let name = dtype_name(dtype);
        let space = match place.kind {
            ClosedPlaceKind::Global { .. } => "device",
            ClosedPlaceKind::Local { kind, .. } => match kind {
                LaunchLocalKind::Workgroup => "threadgroup",
                LaunchLocalKind::Participant | LaunchLocalKind::Register => "device",
            },
        };
        match dtype {
            DType::I32 | DType::U32 => {
                let atomic = if dtype == DType::I32 {
                    "atomic_int"
                } else {
                    "atomic_uint"
                };
                let operation = match op {
                    AtomicOp::Add => "add",
                    AtomicOp::Max => "max",
                    AtomicOp::Min => "min",
                };
                lines.push(format!(
                    "{{ {space} {atomic}* seismic_p = reinterpret_cast<{space} {atomic}*>(&{pointer}[{address}]); \
                     atomic_fetch_{operation}_explicit(seismic_p, {value}, memory_order_relaxed); }}"
                ));
            }
            DType::F32 => {
                let combined = match op {
                    AtomicOp::Add => "f32_add(seismic_a, seismic_v)",
                    AtomicOp::Max => "f32_max(seismic_a, seismic_v)",
                    AtomicOp::Min => "f32_min(seismic_a, seismic_v)",
                };
                lines.push(format!(
                    "{{ {space} atomic_uint* seismic_p = reinterpret_cast<{space} atomic_uint*>(&{pointer}[{address}]); \
                     float seismic_v = float({value}); \
                     uint seismic_expected = atomic_load_explicit(seismic_p, memory_order_relaxed); \
                     while (true) {{ float seismic_a = as_type<float>(seismic_expected); \
                     uint seismic_desired = as_type<uint>({combined}); \
                     if (atomic_compare_exchange_weak_explicit(seismic_p, &seismic_expected, seismic_desired, \
                     memory_order_relaxed, memory_order_relaxed)) break; }} }}"
                ));
            }
            DType::F16 | DType::BF16 => {
                // The 16-bit element lives in the low or high half of an
                // aligned 32-bit word; combine at the element dtype
                // at the element dtype, replace the half-word, then
                // compare-exchange.
                let prefix = if dtype == DType::F16 { "f16" } else { "bf16" };
                let operation = match op {
                    AtomicOp::Add => "add",
                    AtomicOp::Max => "max",
                    AtomicOp::Min => "min",
                };
                let combined = format!("{prefix}_{operation}(seismic_a, seismic_v)");
                lines.push(format!(
                    "{{ {space} char* seismic_byte = reinterpret_cast<{space} char*>(&{pointer}[{address}]); \
                     ulong seismic_bits = ulong(seismic_byte); uint seismic_shift = uint(seismic_bits & 2ul) * 8u; \
                     {space} atomic_uint* seismic_p = reinterpret_cast<{space} atomic_uint*>(seismic_byte - (seismic_bits & 2ul)); \
                     {name} seismic_v = {name}({value}); \
                     uint seismic_expected = atomic_load_explicit(seismic_p, memory_order_relaxed); \
                     while (true) {{ {name} seismic_a = as_type<{name}>(ushort((seismic_expected >> seismic_shift) & 0xffffu)); \
                     uint seismic_new = uint(as_type<ushort>({combined})); \
                     uint seismic_desired = (seismic_expected & ~(0xffffu << seismic_shift)) | (seismic_new << seismic_shift); \
                     if (atomic_compare_exchange_weak_explicit(seismic_p, &seismic_expected, seismic_desired, \
                     memory_order_relaxed, memory_order_relaxed)) break; }} }}"
                ));
            }
            DType::Bool => unreachable!(
                "the typed builder admits atomics on numeric elements only (`AtomicType`)"
            ),
        }
    }

    // -- intrinsics -------------------------------------------------------------------

    fn render_intrinsic(
        &mut self,
        op: &MetalIntrinsic,
        outs: &[ErasedValue],
        args: &[ErasedValue],
        lines: &mut Vec<String>,
    ) {
        match op {
            MetalIntrinsic::LaneIndex => {
                self.assign(outs[0], "int(seismic_lane)".into(), lines);
            }
            MetalIntrinsic::Shuffle { .. } => {
                let expression = format!(
                    "simd_shuffle({}, uint({}))",
                    self.name(args[0]),
                    self.name(args[1])
                );
                self.assign(outs[0], expression, lines);
            }
            MetalIntrinsic::SubgroupReduce { op, .. } => {
                let collective = match op {
                    ReduceOp::Sum => "simd_sum",
                    ReduceOp::Max => "simd_max",
                    ReduceOp::Min => "simd_min",
                    ReduceOp::Argmax => {
                        unreachable!("the registry has no argmax subgroup collective (argmax never reassociates)")
                    }
                };
                let expression = format!("{collective}({})", self.name(args[0]));
                self.assign(outs[0], expression, lines);
            }
            MetalIntrinsic::Matrix {
                left,
                right,
                addend,
                into,
                element,
                accumulator,
                output,
                scratch_left,
                scratch_right,
                scratch_accumulator,
            } => {
                let id = self.temporary_name();
                let rows = self.name(left.extents[0]).to_string();
                let inner = self.name(left.extents[1]).to_string();
                let columns = self.name(right.extents[1]).to_string();
                let elem = dtype_name(*element);
                let acc = dtype_name(*accumulator);
                let out = dtype_name(*output);
                let r = format!("seismic_{id}_r");
                let c = format!("seismic_{id}_c");
                let k = format!("seismic_{id}_k");
                let left_value = self.logical_read_expression(left, &[r.clone(), k.clone()]);
                let right_value = self.logical_read_expression(right, &[k.clone(), c.clone()]);
                let (dp, da) = self.logical_address(into, &[r.clone(), c.clone()]);
                let zero = "0ul".to_string();
                let (sap, saa) = self.logical_address(scratch_left, &[zero.clone(), zero.clone()]);
                let (sbp, sba) = self.logical_address(scratch_right, &[zero.clone(), zero.clone()]);
                let (scp, sca) = self.logical_address(scratch_accumulator, &[zero.clone(), zero]);
                let initial = addend
                    .as_ref()
                    .map(|tensor| self.logical_address(tensor, &[r.clone(), c.clone()]));
                let init = initial
                    .as_ref()
                    .map(|(p, a)| format!("{acc}({p}[{a}])"))
                    .unwrap_or_else(|| format!("{acc}(0)"));
                lines.push(format!(
                    "ulong seismic_{id}_rt = (ulong({rows}) + 7ul) / 8ul; \
                     ulong seismic_{id}_ct = (ulong({columns}) + 7ul) / 8ul; \
                     for (ulong seismic_{id}_tile = ulong(seismic_tg.x); seismic_{id}_tile < seismic_{id}_rt * seismic_{id}_ct; seismic_{id}_tile += ulong(seismic_tgn.x)) {{ \
                     ulong seismic_{id}_tr = (seismic_{id}_tile / seismic_{id}_ct) * 8ul; \
                     ulong seismic_{id}_tc = (seismic_{id}_tile % seismic_{id}_ct) * 8ul; \
                     for (uint seismic_{id}_e = seismic_lane; seismic_{id}_e < 64u; seismic_{id}_e += seismic_simd_width) {{ \
                     ulong {r} = seismic_{id}_tr + ulong(seismic_{id}_e / 8u); ulong {c} = seismic_{id}_tc + ulong(seismic_{id}_e % 8u); \
                     {scp}[{sca} + seismic_{id}_e] = ({r} < ulong({rows}) && {c} < ulong({columns})) ? {init} : {acc}(0); }} \
                     threadgroup_barrier(mem_flags::mem_threadgroup); \
                     simdgroup_matrix<{acc},8,8> seismic_{id}_acc; simdgroup_load(seismic_{id}_acc, {scp} + {sca}, 8); \
                     for (ulong seismic_{id}_kb = 0ul; seismic_{id}_kb < ulong({inner}); seismic_{id}_kb += 8ul) {{ \
                     for (uint seismic_{id}_e = seismic_lane; seismic_{id}_e < 64u; seismic_{id}_e += seismic_simd_width) {{ \
                     ulong {r} = seismic_{id}_tr + ulong(seismic_{id}_e / 8u); ulong {k} = seismic_{id}_kb + ulong(seismic_{id}_e % 8u); \
                     {sap}[{saa} + seismic_{id}_e] = ({r} < ulong({rows}) && {k} < ulong({inner})) ? {elem}({left_value}) : {elem}(0); \
                     {k} = seismic_{id}_kb + ulong(seismic_{id}_e / 8u); ulong {c} = seismic_{id}_tc + ulong(seismic_{id}_e % 8u); \
                     {sbp}[{sba} + seismic_{id}_e] = ({k} < ulong({inner}) && {c} < ulong({columns})) ? {elem}({right_value}) : {elem}(0); }} \
                     threadgroup_barrier(mem_flags::mem_threadgroup); \
                     simdgroup_matrix<{elem},8,8> seismic_{id}_af; simdgroup_matrix<{elem},8,8> seismic_{id}_bf; \
                     simdgroup_matrix<{acc},8,8> seismic_{id}_next; \
                     simdgroup_load(seismic_{id}_af, {sap} + {saa}, 8); simdgroup_load(seismic_{id}_bf, {sbp} + {sba}, 8); \
                     simdgroup_multiply_accumulate(seismic_{id}_next, seismic_{id}_af, seismic_{id}_bf, seismic_{id}_acc); seismic_{id}_acc = seismic_{id}_next; }} \
                     simdgroup_store(seismic_{id}_acc, {scp} + {sca}, 8); threadgroup_barrier(mem_flags::mem_threadgroup); \
                     for (uint seismic_{id}_e = seismic_lane; seismic_{id}_e < 64u; seismic_{id}_e += seismic_simd_width) {{ \
                     ulong {r} = seismic_{id}_tr + ulong(seismic_{id}_e / 8u); ulong {c} = seismic_{id}_tc + ulong(seismic_{id}_e % 8u); \
                     if ({r} < ulong({rows}) && {c} < ulong({columns})) {dp}[{da}] = {out}({scp}[{sca} + seismic_{id}_e]); }} }}"
                ));
            }
        }
    }
}

fn repack_expression(expression: &RepackExpr, source: &str) -> String {
    match expression {
        RepackExpr::SourceBits { bit, width } => {
            let terms = (0..u32::from(*width))
                .map(|offset| {
                    let source_bit = bit + offset;
                    format!(
                        "(((uint({source}[{}ul]) >> {}u) & 1u) << {}u)",
                        source_bit / 8,
                        source_bit % 8,
                        offset
                    )
                })
                .collect::<Vec<_>>();
            if terms.is_empty() {
                "0u".into()
            } else {
                terms.join(" | ")
            }
        }
        RepackExpr::ShiftLeft { value, bits } => {
            format!("(uint({}) << {}u)", repack_expression(value, source), bits)
        }
        RepackExpr::BitOr(left, right) => format!(
            "(uint({}) | uint({}))",
            repack_expression(left, source),
            repack_expression(right, source)
        ),
        RepackExpr::OffsetI32 { value, offset } => {
            format!(
                "(int({}) + int({offset}))",
                repack_expression(value, source)
            )
        }
        RepackExpr::F16ToF32(value) => format!(
            "float(as_type<half>(ushort({})))",
            repack_expression(value, source)
        ),
        RepackExpr::I32ToF32(value) => {
            format!("float(int({}))", repack_expression(value, source))
        }
        RepackExpr::MultiplyF32(left, right) => format!(
            "f32_mul(float({}), float({}))",
            repack_expression(left, source),
            repack_expression(right, source)
        ),
    }
}

fn strict_f32_publication(dtype: DType, expression: &str) -> String {
    match dtype {
        DType::F32 => format!("float({expression})"),
        DType::F16 => format!("f32_to_f16(float({expression}))"),
        DType::BF16 => format!("seismic_bf16_narrow(float({expression}))"),
        DType::I32 => format!("int({expression})"),
        DType::U32 => format!("uint({expression})"),
        DType::Bool => format!("bool({expression})"),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn sum(terms: &[String]) -> String {
    if terms.is_empty() {
        "0ul".into()
    } else {
        terms.join(" + ")
    }
}

fn interpret_code(raw: &str, bits: u32, interpretation: &CodeInterpretation) -> String {
    match interpretation {
        CodeInterpretation::Unsigned => format!("uint({raw})"),
        CodeInterpretation::TwosComplement => {
            let shift = 32 - bits;
            format!("(int(uint({raw}) << {shift}u) >> {shift}u)")
        }
        CodeInterpretation::Offset(offset) => format!("(int({raw}) - {offset})"),
        CodeInterpretation::Table(table) => {
            let mut expression = table[0].to_string();
            for (index, value) in table.iter().copied().enumerate().skip(1) {
                expression = format!("(uint({raw}) == {index}u ? {value} : {expression})");
            }
            expression
        }
    }
}

fn float_code_function(format: FloatCodeFormat) -> &'static str {
    match format {
        FloatCodeFormat::E2M1 => "seismic_decode_e2m1",
        FloatCodeFormat::E4M3 => "seismic_decode_e4m3",
        FloatCodeFormat::UE4M3 => "seismic_decode_ue4m3",
    }
}

pub(crate) fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::Bool => "bool",
        DType::I32 => "int",
        DType::U32 => "uint",
        DType::F16 => "half",
        DType::BF16 => "bfloat",
        DType::F32 => "float",
    }
}

fn value_type_name(ty: &ValueType) -> &'static str {
    match ty {
        ValueType::Scalar(dtype) => dtype_name(*dtype),
        ValueType::Index => "uint",
        ValueType::Bool => "bool",
        ValueType::Vector { .. } => panic!(
            "Metal value-type rendering reached a vector although the target advertises no vector support"
        ),
        ValueType::Opaque { name } => panic!(
            "kernel value of opaque intrinsic type `{name}`: the Metal registrations declare no opaque results"
        ),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Float,
    Int,
    Bool,
}

fn scalar_kind(ty: &ValueType) -> Kind {
    match ty {
        ValueType::Scalar(dtype) if dtype.is_float() => Kind::Float,
        ValueType::Scalar(DType::Bool) | ValueType::Bool => Kind::Bool,
        ValueType::Scalar(_) | ValueType::Index => Kind::Int,
        ValueType::Vector { .. } => {
            value_type_name(ty);
            Kind::Int
        }
        ValueType::Opaque { .. } => {
            value_type_name(ty);
            Kind::Int
        }
    }
}

fn zero_of(ty: &ValueType) -> String {
    match ty {
        ValueType::Bool | ValueType::Scalar(DType::Bool) => "false".into(),
        other => format!("{}(0)", value_type_name(other)),
    }
}

fn constant_expr(value: ConstantValue, ty: &ValueType) -> String {
    let name = value_type_name(ty);
    match value {
        ConstantValue::F32(v) => format!("{name}(as_type<float>({:#010x}u))", v.to_bits()),
        ConstantValue::F16(v) => format!("as_type<half>(ushort({v}u))"),
        ConstantValue::BF16(v) => format!("as_type<bfloat>(ushort({v}u))"),
        ConstantValue::I32(v) => format!("{name}({v})"),
        ConstantValue::U32(v) => format!("{name}({v}u)"),
        ConstantValue::Bool(v) => format!("{v}"),
        ConstantValue::Index(v) => format!("{name}({v}ul)"),
    }
}

fn binary_expr(
    op: BinaryOp,
    ty: &ValueType,
    a: &str,
    b: &str,
    temporary: &mut dyn FnMut() -> String,
) -> String {
    let name = value_type_name(ty);
    let softfloat = match scalar_binary_emission_family(op, ty) {
        Some(
            ScalarEmissionFamily::F32AddSub
            | ScalarEmissionFamily::F32Multiply
            | ScalarEmissionFamily::F32Divide
            | ScalarEmissionFamily::F32Remainder
            | ScalarEmissionFamily::F32MinMax,
        ) => Some("f32"),
        _ => match ty {
            ValueType::Scalar(DType::F16) => Some("f16"),
            ValueType::Scalar(DType::BF16) => Some("bf16"),
            _ => None,
        },
    };
    if let Some(prefix) = softfloat {
        if matches!(op, BinaryOp::Min | BinaryOp::Max) {
            let operation = if op == BinaryOp::Min { "min" } else { "max" };
            return format!("{prefix}_{operation}({a}, {b})");
        }
        let operation = match op {
            BinaryOp::Add => Some("add"),
            BinaryOp::Sub => Some("sub"),
            BinaryOp::Mul => Some("mul"),
            BinaryOp::Div => Some("div"),
            BinaryOp::Rem => Some("rem"),
            BinaryOp::Min | BinaryOp::Max => None,
        };
        if let Some(operation) = operation {
            return format!("{prefix}_{operation}({a}, {b})");
        }
    }
    match (op, scalar_kind(ty)) {
        (BinaryOp::Add, Kind::Float) => format!("{a} + {b}"),
        (BinaryOp::Sub, Kind::Float) => format!("{a} - {b}"),
        (BinaryOp::Mul, Kind::Float) => format!("{a} * {b}"),
        (BinaryOp::Div, Kind::Float) => format!("{a} / {b}"),
        (BinaryOp::Rem, Kind::Float) => format!("fmod({a}, {b})"),
        (BinaryOp::Add, _) => format!("as_type<{name}>(as_type<uint>({a}) + as_type<uint>({b}))"),
        (BinaryOp::Sub, _) => format!("as_type<{name}>(as_type<uint>({a}) - as_type<uint>({b}))"),
        (BinaryOp::Mul, _) => format!("as_type<{name}>(as_type<uint>({a}) * as_type<uint>({b}))"),
        (BinaryOp::Div | BinaryOp::Rem, Kind::Int)
            if matches!(ty, ValueType::Scalar(DType::I32)) =>
        {
            // Euclidean division and remainder (`r` in `0..|rhs|`), exactly
            // the registry semantics; divisor safety is the kernel's `Check`.
            let temp = temporary();
            let remainder = format!(
                "int {temp} = as_type<int>({a}) % as_type<int>({b}); \
                 if ({temp} < 0) {temp} += (as_type<int>({b}) < 0 ? -as_type<int>({b}) : as_type<int>({b}));"
            );
            if op == BinaryOp::Div {
                format!(
                    "({{ {remainder} int(((as_type<int>({a}) - {temp}) / as_type<int>({b}))); }})"
                )
            } else {
                format!("({{ {remainder} {temp}; }})")
            }
        }
        (BinaryOp::Div, _) => format!("{a} / {b}"),
        (BinaryOp::Rem, _) => format!("{a} % {b}"),
        (BinaryOp::Min, _) => format!("min({a}, {b})"),
        (BinaryOp::Max, _) => format!("max({a}, {b})"),
    }
}

fn fma_function(ty: &ValueType) -> &'static str {
    match scalar_fma_emission_family(ty) {
        Some(ScalarEmissionFamily::F32FusedMultiplyAdd) => "f32_mulAdd",
        _ => match ty {
            ValueType::Scalar(DType::F16) => "f16_mulAdd",
            ValueType::Scalar(DType::BF16) => "bf16_mulAdd",
            _ => "fma",
        },
    }
}

fn float_sign_op(ty: &ValueType, value: &str, negate: bool) -> String {
    match ty {
        ValueType::Scalar(DType::F32) => {
            let op = if negate {
                "^ 0x80000000u"
            } else {
                "& 0x7fffffffu"
            };
            format!("as_type<float>(as_type<uint>({value}) {op})")
        }
        ValueType::Scalar(DType::F16) => {
            let op = if negate { "^ 0x8000u" } else { "& 0x7fffu" };
            format!("as_type<half>(ushort(as_type<ushort>({value}) {op}))")
        }
        ValueType::Scalar(DType::BF16) => {
            let op = if negate { "^ 0x8000u" } else { "& 0x7fffu" };
            format!("as_type<bfloat>(ushort(as_type<ushort>({value}) {op}))")
        }
        _ => unreachable!("closed float unary operation has a floating scalar type"),
    }
}

fn math_expr(op: MathOp, ty: &ValueType, a: &str) -> String {
    let namespace = "fast";
    let function = match op {
        MathOp::Exp | MathOp::ExpFast => "exp",
        MathOp::Rsqrt => "rsqrt",
        MathOp::Sqrt => "sqrt",
        MathOp::Log => "log",
        MathOp::Sin => "sin",
        MathOp::Cos => "cos",
        MathOp::Abs => {
            return match scalar_kind(ty) {
                Kind::Float => format!("fabs({a})"),
                Kind::Int | Kind::Bool => format!("abs({a})"),
            }
        }
        MathOp::Fma | MathOp::Max | MathOp::Min => unreachable!(
            "`{:?}` is a multi-operand op the builder lowers through `Fma`/`Binary`, never `Math`",
            op
        ),
    };
    match ty {
        // Half-precision widening: transcendentals are evaluated in `float`
        // and rounded once to the element type.
        ValueType::Scalar(DType::F16) => format!("half({namespace}::{function}(float({a})))"),
        ValueType::Scalar(DType::BF16) => format!("bfloat({namespace}::{function}(float({a})))"),
        _ => format!("{namespace}::{function}({a})"),
    }
}

fn cast_expr(from: &ValueType, to: &ValueType, a: &str) -> String {
    let from_kind = scalar_kind(from);
    let to_name = value_type_name(to);
    let emission = scalar_conversion_emission_family(from, to);
    match emission {
        Some(ScalarEmissionFamily::F32ToF16) => return format!("f32_to_f16({a})"),
        Some(ScalarEmissionFamily::F16ToF32) => return format!("f16_to_f32({a})"),
        Some(ScalarEmissionFamily::F32ToBF16) => return format!("seismic_bf16_narrow({a})"),
        Some(ScalarEmissionFamily::BF16ToF32) => return format!("seismic_bf16_widen({a})"),
        Some(ScalarEmissionFamily::F32ToInteger) => {
            return match to {
                ValueType::Scalar(DType::I32) => {
                    format!("int(clamp(trunc(float({a})), -2147483648.0f, 2147483647.0f))")
                }
                ValueType::Index | ValueType::Scalar(DType::U32) => {
                    format!("uint(clamp(trunc(float({a})), 0.0f, 4294967295.0f))")
                }
                _ => unreachable!("F32-to-integer emission has an integer destination"),
            }
        }
        Some(ScalarEmissionFamily::IntegerToF32) => return format!("{to_name}({a})"),
        _ => {}
    }
    match (from, to) {
        (ValueType::Scalar(DType::F16), ValueType::Scalar(DType::BF16)) => {
            return format!("seismic_bf16_narrow(f16_to_f32({a}))")
        }
        (ValueType::Scalar(DType::BF16), ValueType::Scalar(DType::F16)) => {
            return format!("f32_to_f16(seismic_bf16_widen({a}))")
        }
        _ => {}
    }
    match (from_kind, to) {
        (_, ValueType::Bool) | (_, ValueType::Scalar(DType::Bool)) => format!("({a} != 0)"),
        (
            Kind::Int,
            ValueType::Index | ValueType::Scalar(DType::I32) | ValueType::Scalar(DType::U32),
        ) => {
            // Integer-to-integer casts preserve the low 32 bits.
            format!("as_type<{to_name}>(as_type<uint>({a}))")
        }
        (Kind::Bool, _) => format!("{to_name}({a})"),
        (Kind::Float, ValueType::Scalar(DType::I32)) => {
            format!("int(clamp(trunc(float({a})), -2147483648.0f, 2147483647.0f))")
        }
        (Kind::Float, ValueType::Index | ValueType::Scalar(DType::U32)) => {
            format!("uint(clamp(trunc(float({a})), 0.0f, 4294967295.0f))")
        }
        _ => format!("{to_name}({a})"),
    }
}

fn cmp_symbol(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Eq => "==",
        CmpOp::Ne => "!=",
        CmpOp::Lt => "<",
        CmpOp::Le => "<=",
        CmpOp::Gt => ">",
        CmpOp::Ge => ">=",
    }
}

fn compare_expr(op: CmpOp, ty: &ValueType, a: &str, b: &str) -> String {
    let prefix = match scalar_comparison_emission_family(ty) {
        Some(ScalarEmissionFamily::F32Comparison) => Some("f32"),
        _ => match ty {
            ValueType::Scalar(DType::F16) => Some("f16"),
            ValueType::Scalar(DType::BF16) => Some("bf16"),
            _ => None,
        },
    };
    let Some(prefix) = prefix else {
        return format!("({a} {} {b})", cmp_symbol(op));
    };
    match op {
        CmpOp::Eq => format!("{prefix}_eq({a}, {b})"),
        CmpOp::Ne => format!("!{prefix}_eq({a}, {b})"),
        CmpOp::Lt => format!("{prefix}_lt({a}, {b})"),
        CmpOp::Le => format!(
            "!{prefix}_lt({b}, {a}) && !seismic_{prefix}_nan({a}) && !seismic_{prefix}_nan({b})"
        ),
        CmpOp::Gt => format!("{prefix}_lt({b}, {a})"),
        CmpOp::Ge => format!(
            "!{prefix}_lt({a}, {b}) && !seismic_{prefix}_nan({a}) && !seismic_{prefix}_nan({b})"
        ),
    }
}

fn axis_component(axis: u8) -> &'static str {
    match axis {
        0 => "x",
        1 => "y",
        _ => "z",
    }
}

fn geometry_expr(kind: GeometryValue) -> String {
    match kind {
        GeometryValue::WorkgroupId(axis) => format!("seismic_tg.{}", axis_component(axis)),
        GeometryValue::LocalId(axis) => format!("seismic_tid.{}", axis_component(axis)),
        GeometryValue::GlobalId(axis) => {
            let c = axis_component(axis);
            format!("(seismic_tg.{c} * seismic_tpg.{c} + seismic_tid.{c})")
        }
        GeometryValue::WorkgroupSize(axis) => format!("seismic_tpg.{}", axis_component(axis)),
        GeometryValue::GridSize(axis) => format!("seismic_tgn.{}", axis_component(axis)),
        GeometryValue::SubgroupLane => "seismic_lane".into(),
    }
}

/// Decodes one 64-bit parameter word into a scalar of `dtype`.
fn decode_word(dtype: DType, word: &str) -> String {
    match dtype {
        DType::F32 => format!("as_type<float>(uint({word}))"),
        DType::F16 => format!("as_type<half>(ushort({word}))"),
        DType::BF16 => format!("as_type<bfloat>(ushort({word}))"),
        DType::I32 => format!("as_type<int>(uint({word}))"),
        DType::U32 => format!("uint({word})"),
        DType::Bool => format!("({word} != 0ul)"),
    }
}

/// Encodes a scalar of `dtype` into the 32-bit side word.
fn encode_word(dtype: DType, value: &str) -> String {
    match dtype {
        DType::F32 => format!("as_type<uint>({value})"),
        DType::F16 => format!("uint(as_type<ushort>({value}))"),
        DType::BF16 => format!("uint(as_type<ushort>({value}))"),
        DType::I32 => format!("as_type<uint>({value})"),
        DType::U32 => format!("uint({value})"),
        DType::Bool => format!("uint({value})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_emission_descriptors_are_exact() {
        let f32_ty = ValueType::Scalar(DType::F32);
        for op in [BinaryOp::Add, BinaryOp::Sub] {
            assert_eq!(
                scalar_binary_emission_family(op, &f32_ty),
                Some(ScalarEmissionFamily::F32AddSub)
            );
        }
        for (op, family) in [
            (BinaryOp::Mul, ScalarEmissionFamily::F32Multiply),
            (BinaryOp::Div, ScalarEmissionFamily::F32Divide),
            (BinaryOp::Rem, ScalarEmissionFamily::F32Remainder),
            (BinaryOp::Min, ScalarEmissionFamily::F32MinMax),
            (BinaryOp::Max, ScalarEmissionFamily::F32MinMax),
        ] {
            assert_eq!(scalar_binary_emission_family(op, &f32_ty), Some(family));
        }
        assert_eq!(
            scalar_fma_emission_family(&f32_ty),
            Some(ScalarEmissionFamily::F32FusedMultiplyAdd)
        );
        assert_eq!(
            scalar_comparison_emission_family(&f32_ty),
            Some(ScalarEmissionFamily::F32Comparison)
        );

        let f16_ty = ValueType::Scalar(DType::F16);
        let bf16_ty = ValueType::Scalar(DType::BF16);
        let i32_ty = ValueType::Scalar(DType::I32);
        let u32_ty = ValueType::Scalar(DType::U32);
        for (from, to, family) in [
            (&f32_ty, &f16_ty, ScalarEmissionFamily::F32ToF16),
            (&f16_ty, &f32_ty, ScalarEmissionFamily::F16ToF32),
            (&f32_ty, &bf16_ty, ScalarEmissionFamily::F32ToBF16),
            (&bf16_ty, &f32_ty, ScalarEmissionFamily::BF16ToF32),
            (&f32_ty, &i32_ty, ScalarEmissionFamily::F32ToInteger),
            (&u32_ty, &f32_ty, ScalarEmissionFamily::IntegerToF32),
        ] {
            assert_eq!(scalar_conversion_emission_family(from, to), Some(family));
        }
        assert_eq!(
            scalar_binary_emission_family(BinaryOp::Add, &i32_ty),
            Some(ScalarEmissionFamily::NativeIntegerBit)
        );
        assert_eq!(
            scalar_integer_bit_emission_family(&u32_ty),
            Some(ScalarEmissionFamily::NativeIntegerBit)
        );
        assert_eq!(
            scalar_control_emission_family(),
            ScalarEmissionFamily::NativeControl
        );
        assert_eq!(
            scalar_comparison_emission_family(&ValueType::Bool),
            Some(ScalarEmissionFamily::NativeControl)
        );
        assert_eq!(scalar_fma_emission_family(&f16_ty), None);
    }

    #[test]
    fn descriptor_driven_helpers_preserve_msl_spelling() {
        let f32_ty = ValueType::Scalar(DType::F32);
        let mut temporary = || "temporary".to_owned();
        for (op, expected) in [
            (BinaryOp::Add, "f32_add(a, b)"),
            (BinaryOp::Sub, "f32_sub(a, b)"),
            (BinaryOp::Mul, "f32_mul(a, b)"),
            (BinaryOp::Div, "f32_div(a, b)"),
            (BinaryOp::Rem, "f32_rem(a, b)"),
            (BinaryOp::Min, "f32_min(a, b)"),
            (BinaryOp::Max, "f32_max(a, b)"),
        ] {
            assert_eq!(binary_expr(op, &f32_ty, "a", "b", &mut temporary), expected);
        }
        assert_eq!(fma_function(&f32_ty), "f32_mulAdd");
        assert_eq!(compare_expr(CmpOp::Eq, &f32_ty, "a", "b"), "f32_eq(a, b)");
        assert_eq!(compare_expr(CmpOp::Ne, &f32_ty, "a", "b"), "!f32_eq(a, b)");
        assert_eq!(compare_expr(CmpOp::Lt, &f32_ty, "a", "b"), "f32_lt(a, b)");
        assert_eq!(compare_expr(CmpOp::Gt, &f32_ty, "a", "b"), "f32_lt(b, a)");

        let f16_ty = ValueType::Scalar(DType::F16);
        let bf16_ty = ValueType::Scalar(DType::BF16);
        let i32_ty = ValueType::Scalar(DType::I32);
        let u32_ty = ValueType::Scalar(DType::U32);
        assert_eq!(cast_expr(&f32_ty, &f16_ty, "a"), "f32_to_f16(a)");
        assert_eq!(cast_expr(&f16_ty, &f32_ty, "a"), "f16_to_f32(a)");
        assert_eq!(cast_expr(&f32_ty, &bf16_ty, "a"), "seismic_bf16_narrow(a)");
        assert_eq!(cast_expr(&bf16_ty, &f32_ty, "a"), "seismic_bf16_widen(a)");
        assert_eq!(
            cast_expr(&f32_ty, &i32_ty, "a"),
            "int(clamp(trunc(float(a)), -2147483648.0f, 2147483647.0f))"
        );
        assert_eq!(cast_expr(&u32_ty, &f32_ty, "a"), "float(a)");
    }
}
