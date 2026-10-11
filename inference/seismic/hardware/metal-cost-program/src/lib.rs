//! Controlled proof of a shared target-closed Metal cost program.
//!
//! This is deliberately outside production.  `ClosedCostProgram` has no
//! unknown/gap variant: construction either returns a program which both
//! fixtures can interpret, or one aggregate `ConstructionFailure`.

use seismic_ir::metal::ScalarEmissionFamily;

const SOFTFLOAT_SOURCE: &str =
    include_str!("../../../../seismic/backends/metal/src/softfloat.metal");

/// Facts which are inputs to closure, never estimator guesses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClosureEvidence {
    pub native_realization_bounds: bool,
    pub invocation_shape: bool,
    pub path_cohorts: bool,
    pub address_relations: bool,
    pub strict_f32_multiply_path: Option<StrictMultiplyPath>,
    /// `None` describes the current weak-CAS implementation: Metal gives no
    /// deterministic finite retry bound.
    pub weak_cas_retry_bound: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrictMultiplyPath {
    /// Normal operands/result, exact discarded bits, high product bit set.
    FiniteNormalExact,
    /// The same path with the product-normalization shift taken.
    FiniteNormalExactShiftProduct,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepresentativeMechanism {
    ScheduleAndResidency,
    GlobalMemory,
    NativeIntegerAtomic,
    WeakCasFloatAtomic,
    SubgroupCollective,
    SimdgroupMatrix,
}

/// Every failure names the missing *semantic input*.  There is no generic
/// unsupported/unassessed bucket.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Obligation {
    pub subject: Subject,
    pub kind: ObligationKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Subject {
    Scalar(ScalarEmissionFamily),
    Mechanism(RepresentativeMechanism),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObligationKind {
    /// The current family merged operations which render differently, such as
    /// add/sub, min/max, comparison predicates, or integer/bit operations.
    PreserveOperationDiscriminant,
    /// The renderer expands a helper after the present estimator input.  The
    /// expansion must move into the shared program.
    PreserveExpandedHelperProgram,
    /// Occupancy/spill/scheduling bounds depend on native realization.
    SupplyNativeRealizationBounds,
    /// Launch/cohort/staging cardinalities are required.
    SupplyInvocationShape,
    /// Runtime values select helper paths and SIMD divergence cohorts.
    SupplyPathCohorts,
    /// Transactions, reuse and contention depend on address equivalence.
    SupplyAddressRelations,
    /// Current weak compare-exchange has no finite deterministic retry bound.
    ReplaceOrContractWeakCasProgress,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstructionFailure {
    pub obligations: Vec<Obligation>,
}

/// Machine-facing primitives. Strict floating point is intentionally absent:
/// helper routines decompose into these physical mechanisms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PhysicalPrimitive {
    Bitcast32,
    U32And,
    U32Or,
    U32Xor,
    U32Shift,
    U32AddSub,
    U64Multiply,
    U64Shift,
    IntegerConvert,
    IntegerCompare,
    ControlTransfer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrimitiveDemand {
    pub primitive: PhysicalPrimitive,
    pub count: u32,
}

/// One authoritative executable node. Rendering and demand extraction both
/// exhaustively interpret this enum; no node carries a parallel cost label.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PhysicalNode {
    BitcastF32ToU32 {
        dst: &'static str,
        src: &'static str,
    },
    BitcastU32ToF32 {
        dst: &'static str,
        src: &'static str,
    },
    U32AndImm {
        dst: &'static str,
        src: &'static str,
        imm: u32,
    },
    U32OrImm {
        dst: &'static str,
        src: &'static str,
        imm: u32,
    },
    U32Or {
        dst: &'static str,
        left: &'static str,
        right: &'static str,
    },
    U32Xor {
        dst: &'static str,
        left: &'static str,
        right: &'static str,
    },
    U32Add {
        dst: &'static str,
        left: &'static str,
        right: &'static str,
    },
    U32AddImm {
        dst: &'static str,
        src: &'static str,
        imm: u32,
    },
    U32SubImm {
        dst: &'static str,
        src: &'static str,
        imm: u32,
    },
    U32ShlImm {
        dst: &'static str,
        src: &'static str,
        imm: u32,
    },
    U32ShrImm {
        dst: &'static str,
        src: &'static str,
        imm: u32,
    },
    U64Multiply {
        dst: &'static str,
        left: &'static str,
        right: &'static str,
    },
    U64ShlImm {
        dst: &'static str,
        src: &'static str,
        imm: u32,
    },
    U64ShrImm {
        dst: &'static str,
        src: &'static str,
        imm: u32,
    },
    U64NeZeroToU32 {
        dst: &'static str,
        src: &'static str,
    },
    U64ToU32 {
        dst: &'static str,
        src: &'static str,
    },
    GuardU32EqImm {
        src: &'static str,
        imm: u32,
    },
    GuardU32NeImm {
        src: &'static str,
        imm: u32,
    },
    GuardU32LtImm {
        src: &'static str,
        imm: u32,
    },
    GuardU32GeImm {
        src: &'static str,
        imm: u32,
    },
    ReturnU32 {
        src: &'static str,
    },
    ReturnF32 {
        src: &'static str,
    },
}

/// A sealed helper or native operation body. The body itself is the shared
/// cost program; it is not a renderer recipe paired with estimator metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosedOperation {
    name: &'static str,
    parameters: &'static str,
    result_type: &'static str,
    nodes: Vec<PhysicalNode>,
}

/// The only successful construction product. There is no `Gap`, `Unknown`,
/// optional program, fallback service, or zero-demand sentinel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClosedCostProgram {
    operations: Vec<ClosedOperation>,
}

impl ClosedCostProgram {
    pub fn operations(&self) -> &[ClosedOperation] {
        &self.operations
    }
}

/// Fixture renderer: emits the same physical nodes the estimator traverses.
pub fn render_fixture(program: &ClosedCostProgram) -> String {
    let mut rendered = String::new();
    for operation in &program.operations {
        rendered.push_str(operation.result_type);
        rendered.push(' ');
        rendered.push_str(operation.name);
        rendered.push('(');
        rendered.push_str(operation.parameters);
        rendered.push_str(") {\n");
        for node in &operation.nodes {
            rendered.push_str("  ");
            rendered.push_str(&render_node(node));
            rendered.push('\n');
        }
        rendered.push_str("}\n");
    }
    rendered
}

/// Fixture estimator transfer: it consumes the exact same operations and
/// merely sums already-expanded primitive demand.
pub fn demand_fixture(program: &ClosedCostProgram) -> Vec<PrimitiveDemand> {
    use std::collections::BTreeMap;
    let mut sums = BTreeMap::new();
    for operation in &program.operations {
        for node in &operation.nodes {
            for primitive in node_primitives(node) {
                *sums.entry(*primitive).or_insert(0u32) += 1;
            }
        }
    }
    sums.into_iter()
        .map(|(primitive, count)| PrimitiveDemand { primitive, count })
        .collect()
}

/// A closed native arithmetic fixture. This is what production construction
/// should produce after preserving the exact operation, not merely the broad
/// `NativeIntegerBit` family.
pub fn native_u32_add() -> ClosedOperation {
    ClosedOperation {
        name: "proof_native_u32_add",
        parameters: "uint a, uint b",
        result_type: "uint",
        nodes: vec![
            PhysicalNode::U32Add {
                dst: "sum",
                left: "a",
                right: "b",
            },
            PhysicalNode::ReturnU32 { src: "sum" },
        ],
    }
}

/// Actual finite-normal path of the repository's `f32_mul` helper.
///
/// This is a path-specialized executable expansion, not a call paired with a
/// manually maintained count vector. It follows the repository helper through
/// decode, guards, significand multiplication, jam shift, normalization and
/// the normal/exact round-pack path.
pub fn strict_f32_multiply(path: StrictMultiplyPath) -> ClosedOperation {
    assert!(SOFTFLOAT_SOURCE.contains("float32_t f32_mul( float32_t a, float32_t b )"));
    assert!(SOFTFLOAT_SOURCE
        .contains("sigZ = softfloat_shortShiftRightJam64( (uint_fast64_t) sigA * sigB, 32 );"));
    assert!(SOFTFLOAT_SOURCE.contains("sig = (sig + roundIncrement)>>7;"));
    let shift_product = match path {
        StrictMultiplyPath::FiniteNormalExact => false,
        StrictMultiplyPath::FiniteNormalExactShiftProduct => true,
    };
    let mut nodes = vec![
        PhysicalNode::BitcastF32ToU32 {
            dst: "ui_a",
            src: "a",
        },
        PhysicalNode::BitcastF32ToU32 {
            dst: "ui_b",
            src: "b",
        },
        PhysicalNode::U32ShrImm {
            dst: "sign_a",
            src: "ui_a",
            imm: 31,
        },
        PhysicalNode::U32ShrImm {
            dst: "exp_a_raw",
            src: "ui_a",
            imm: 23,
        },
        PhysicalNode::U32AndImm {
            dst: "exp_a",
            src: "exp_a_raw",
            imm: 0xff,
        },
        PhysicalNode::U32AndImm {
            dst: "sig_a",
            src: "ui_a",
            imm: 0x007f_ffff,
        },
        PhysicalNode::U32ShrImm {
            dst: "sign_b",
            src: "ui_b",
            imm: 31,
        },
        PhysicalNode::U32ShrImm {
            dst: "exp_b_raw",
            src: "ui_b",
            imm: 23,
        },
        PhysicalNode::U32AndImm {
            dst: "exp_b",
            src: "exp_b_raw",
            imm: 0xff,
        },
        PhysicalNode::U32AndImm {
            dst: "sig_b",
            src: "ui_b",
            imm: 0x007f_ffff,
        },
        PhysicalNode::U32Xor {
            dst: "sign_z",
            left: "sign_a",
            right: "sign_b",
        },
        PhysicalNode::GuardU32NeImm {
            src: "exp_a",
            imm: 0xff,
        },
        PhysicalNode::GuardU32NeImm {
            src: "exp_b",
            imm: 0xff,
        },
        PhysicalNode::GuardU32NeImm {
            src: "exp_a",
            imm: 0,
        },
        PhysicalNode::GuardU32NeImm {
            src: "exp_b",
            imm: 0,
        },
        PhysicalNode::U32Add {
            dst: "exp_sum",
            left: "exp_a",
            right: "exp_b",
        },
        PhysicalNode::GuardU32GeImm {
            src: "exp_sum",
            imm: 0x7f,
        },
        PhysicalNode::U32SubImm {
            dst: "exp_z",
            src: "exp_sum",
            imm: 0x7f,
        },
        PhysicalNode::U32OrImm {
            dst: "sig_a_hidden",
            src: "sig_a",
            imm: 0x0080_0000,
        },
        PhysicalNode::U32ShlImm {
            dst: "sig_a_scaled",
            src: "sig_a_hidden",
            imm: 7,
        },
        PhysicalNode::U32OrImm {
            dst: "sig_b_hidden",
            src: "sig_b",
            imm: 0x0080_0000,
        },
        PhysicalNode::U32ShlImm {
            dst: "sig_b_scaled",
            src: "sig_b_hidden",
            imm: 8,
        },
        PhysicalNode::U64Multiply {
            dst: "product",
            left: "sig_a_scaled",
            right: "sig_b_scaled",
        },
        // Exact expansion of `softfloat_shortShiftRightJam64(product, 32)`.
        PhysicalNode::U64ShrImm {
            dst: "product_high",
            src: "product",
            imm: 32,
        },
        PhysicalNode::U64ShlImm {
            dst: "product_discarded",
            src: "product",
            imm: 32,
        },
        PhysicalNode::U64NeZeroToU32 {
            dst: "product_jam",
            src: "product_discarded",
        },
        PhysicalNode::U64ToU32 {
            dst: "product_high_u32",
            src: "product_high",
        },
        PhysicalNode::U32Or {
            dst: "sig_z",
            left: "product_high_u32",
            right: "product_jam",
        },
    ];
    if shift_product {
        nodes.extend([
            PhysicalNode::GuardU32LtImm {
                src: "sig_z",
                imm: 0x4000_0000,
            },
            PhysicalNode::U32SubImm {
                dst: "exp_z_normal",
                src: "exp_z",
                imm: 1,
            },
            PhysicalNode::U32ShlImm {
                dst: "sig_z_normal",
                src: "sig_z",
                imm: 1,
            },
        ]);
    } else {
        nodes.push(PhysicalNode::GuardU32GeImm {
            src: "sig_z",
            imm: 0x4000_0000,
        });
    }
    let exp = if shift_product {
        "exp_z_normal"
    } else {
        "exp_z"
    };
    let sig = if shift_product {
        "sig_z_normal"
    } else {
        "sig_z"
    };
    nodes.extend([
        PhysicalNode::U32AndImm {
            dst: "round_bits",
            src: sig,
            imm: 0x7f,
        },
        PhysicalNode::GuardU32EqImm {
            src: "round_bits",
            imm: 0,
        },
        PhysicalNode::GuardU32LtImm {
            src: exp,
            imm: 0xfd,
        },
        PhysicalNode::U32AddImm {
            dst: "rounded_pre_shift",
            src: sig,
            imm: 0x40,
        },
        PhysicalNode::U32ShrImm {
            dst: "rounded_sig",
            src: "rounded_pre_shift",
            imm: 7,
        },
        PhysicalNode::U32ShlImm {
            dst: "packed_sign",
            src: "sign_z",
            imm: 31,
        },
        PhysicalNode::U32ShlImm {
            dst: "packed_exp",
            src: exp,
            imm: 23,
        },
        PhysicalNode::U32Or {
            dst: "sign_exp",
            left: "packed_sign",
            right: "packed_exp",
        },
        PhysicalNode::U32Or {
            dst: "ui_z",
            left: "sign_exp",
            right: "rounded_sig",
        },
        PhysicalNode::BitcastU32ToF32 {
            dst: "z",
            src: "ui_z",
        },
        PhysicalNode::ReturnF32 { src: "z" },
    ]);
    ClosedOperation {
        name: "proof_f32_mul_finite_normal_exact",
        parameters: "float a, float b",
        result_type: "float",
        nodes,
    }
}

pub fn fixture_program(path: StrictMultiplyPath) -> ClosedCostProgram {
    ClosedCostProgram {
        operations: vec![native_u32_add(), strict_f32_multiply(path)],
    }
}

fn node_primitives(node: &PhysicalNode) -> &'static [PhysicalPrimitive] {
    use PhysicalNode as N;
    use PhysicalPrimitive as P;
    match node {
        N::BitcastF32ToU32 { .. } | N::BitcastU32ToF32 { .. } => &[P::Bitcast32],
        N::U32AndImm { .. } => &[P::U32And],
        N::U32OrImm { .. } | N::U32Or { .. } => &[P::U32Or],
        N::U32Xor { .. } => &[P::U32Xor],
        N::U32Add { .. } | N::U32AddImm { .. } | N::U32SubImm { .. } => &[P::U32AddSub],
        N::U32ShlImm { .. } | N::U32ShrImm { .. } => &[P::U32Shift],
        N::U64Multiply { .. } => &[P::U64Multiply],
        N::U64ShlImm { .. } | N::U64ShrImm { .. } => &[P::U64Shift],
        N::U64NeZeroToU32 { .. } => &[P::IntegerCompare, P::IntegerConvert],
        N::U64ToU32 { .. } => &[P::IntegerConvert],
        N::GuardU32EqImm { .. }
        | N::GuardU32NeImm { .. }
        | N::GuardU32LtImm { .. }
        | N::GuardU32GeImm { .. } => &[P::IntegerCompare, P::ControlTransfer],
        N::ReturnU32 { .. } | N::ReturnF32 { .. } => &[P::ControlTransfer],
    }
}

fn render_node(node: &PhysicalNode) -> String {
    use PhysicalNode as N;
    match node {
        N::BitcastF32ToU32 { dst, src } => format!("uint {dst} = as_type<uint>({src});"),
        N::BitcastU32ToF32 { dst, src } => format!("float {dst} = as_type<float>({src});"),
        N::U32AndImm { dst, src, imm } => format!("uint {dst} = {src} & {imm:#010x}u;"),
        N::U32OrImm { dst, src, imm } => format!("uint {dst} = {src} | {imm:#010x}u;"),
        N::U32Or { dst, left, right } => format!("uint {dst} = {left} | {right};"),
        N::U32Xor { dst, left, right } => format!("uint {dst} = {left} ^ {right};"),
        N::U32Add { dst, left, right } => format!("uint {dst} = {left} + {right};"),
        N::U32AddImm { dst, src, imm } => format!("uint {dst} = {src} + {imm}u;"),
        N::U32SubImm { dst, src, imm } => format!("uint {dst} = {src} - {imm}u;"),
        N::U32ShlImm { dst, src, imm } => format!("uint {dst} = {src} << {imm}u;"),
        N::U32ShrImm { dst, src, imm } => format!("uint {dst} = {src} >> {imm}u;"),
        N::U64Multiply { dst, left, right } => {
            format!("ulong {dst} = ulong({left}) * ulong({right});")
        }
        N::U64ShlImm { dst, src, imm } => format!("ulong {dst} = {src} << {imm}u;"),
        N::U64ShrImm { dst, src, imm } => format!("ulong {dst} = {src} >> {imm}u;"),
        N::U64NeZeroToU32 { dst, src } => format!("uint {dst} = uint({src} != 0ul);"),
        N::U64ToU32 { dst, src } => format!("uint {dst} = uint({src});"),
        N::GuardU32EqImm { src, imm } => {
            format!("if ({src} != {imm:#010x}u) return as_type<float>(0x7fc00000u);")
        }
        N::GuardU32NeImm { src, imm } => {
            format!("if ({src} == {imm:#010x}u) return as_type<float>(0x7fc00000u);")
        }
        N::GuardU32LtImm { src, imm } => {
            format!("if (!({src} < {imm:#010x}u)) return as_type<float>(0x7fc00000u);")
        }
        N::GuardU32GeImm { src, imm } => {
            format!("if (!({src} >= {imm:#010x}u)) return as_type<float>(0x7fc00000u);")
        }
        N::ReturnU32 { src } | N::ReturnF32 { src } => format!("return {src};"),
    }
}

/// Audit one member of the current production scalar family. The exhaustive
/// match is the compile-time coverage proof: adding a production variant makes
/// this crate fail to compile until it is deliberately classified.
pub fn audit_scalar_family(
    family: ScalarEmissionFamily,
    evidence: ClosureEvidence,
) -> Vec<ObligationKind> {
    use ObligationKind as O;
    use ScalarEmissionFamily as F;
    match family {
        F::F32AddSub | F::F32MinMax | F::F32Comparison | F::NativeIntegerBit => {
            vec![
                O::PreserveOperationDiscriminant,
                O::PreserveExpandedHelperProgram,
            ]
        }
        F::F32Multiply => {
            let mut obligations = Vec::new();
            if evidence.strict_f32_multiply_path.is_none() {
                obligations.push(O::SupplyPathCohorts);
            }
            obligations
        }
        F::F32Divide
        | F::F32Remainder
        | F::F32FusedMultiplyAdd
        | F::F32ToF16
        | F::F16ToF32
        | F::F32ToBF16 => vec![O::PreserveExpandedHelperProgram, O::SupplyPathCohorts],
        F::F32ToInteger | F::IntegerToF32 | F::NativeControl => {
            vec![O::PreserveOperationDiscriminant]
        }
        F::BF16ToF32 => Vec::new(),
    }
}

/// Audit representative non-scalar mechanisms which the shared program must
/// eventually own. This is also exhaustive over the proof's closed mechanism
/// vocabulary.
pub fn audit_mechanism(
    mechanism: RepresentativeMechanism,
    evidence: ClosureEvidence,
) -> Vec<ObligationKind> {
    use ObligationKind as O;
    use RepresentativeMechanism as M;
    let mut obligations = Vec::new();
    match mechanism {
        M::ScheduleAndResidency => {
            if !evidence.native_realization_bounds {
                obligations.push(O::SupplyNativeRealizationBounds);
            }
            if !evidence.invocation_shape {
                obligations.push(O::SupplyInvocationShape);
            }
        }
        M::GlobalMemory | M::NativeIntegerAtomic => {
            if !evidence.invocation_shape {
                obligations.push(O::SupplyInvocationShape);
            }
            if !evidence.address_relations {
                obligations.push(O::SupplyAddressRelations);
            }
        }
        M::WeakCasFloatAtomic => {
            if !evidence.address_relations {
                obligations.push(O::SupplyAddressRelations);
            }
            if evidence.weak_cas_retry_bound.is_none() {
                obligations.push(O::ReplaceOrContractWeakCasProgress);
            }
        }
        M::SubgroupCollective => {
            if !evidence.invocation_shape {
                obligations.push(O::SupplyInvocationShape);
            }
            if !evidence.path_cohorts {
                obligations.push(O::SupplyPathCohorts);
            }
        }
        M::SimdgroupMatrix => {
            if !evidence.native_realization_bounds {
                obligations.push(O::SupplyNativeRealizationBounds);
            }
            if !evidence.invocation_shape {
                obligations.push(O::SupplyInvocationShape);
            }
            if !evidence.address_relations {
                obligations.push(O::SupplyAddressRelations);
            }
        }
    }
    obligations
}

/// Whole-program gate. Every obligation is retained; there is no early return.
pub fn construct_current_vocabulary(
    evidence: ClosureEvidence,
) -> Result<ClosedCostProgram, ConstructionFailure> {
    let mut obligations = Vec::new();
    audit_every_scalar(evidence, &mut obligations);
    audit_every_mechanism(evidence, &mut obligations);
    if obligations.is_empty() {
        Ok(fixture_program(
            evidence
                .strict_f32_multiply_path
                .expect("empty obligations prove the path fact exists"),
        ))
    } else {
        Err(ConstructionFailure { obligations })
    }
}

fn record(subject: Subject, kinds: Vec<ObligationKind>, obligations: &mut Vec<Obligation>) {
    obligations.extend(kinds.into_iter().map(|kind| Obligation { subject, kind }));
}

fn audit_every_scalar(evidence: ClosureEvidence, obligations: &mut Vec<Obligation>) {
    use ScalarEmissionFamily as F;
    // This is construction, not a comparison with a second expected list.
    for family in [
        F::F32AddSub,
        F::F32Multiply,
        F::F32Divide,
        F::F32Remainder,
        F::F32MinMax,
        F::F32FusedMultiplyAdd,
        F::F32Comparison,
        F::F32ToF16,
        F::F16ToF32,
        F::F32ToBF16,
        F::BF16ToF32,
        F::F32ToInteger,
        F::IntegerToF32,
        F::NativeControl,
        F::NativeIntegerBit,
    ] {
        record(
            Subject::Scalar(family),
            audit_scalar_family(family, evidence),
            obligations,
        );
    }
}

fn audit_every_mechanism(evidence: ClosureEvidence, obligations: &mut Vec<Obligation>) {
    use RepresentativeMechanism as M;
    for mechanism in [
        M::ScheduleAndResidency,
        M::GlobalMemory,
        M::NativeIntegerAtomic,
        M::WeakCasFloatAtomic,
        M::SubgroupCollective,
        M::SimdgroupMatrix,
    ] {
        record(
            Subject::Mechanism(mechanism),
            audit_mechanism(mechanism, evidence),
            obligations,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_shared_program_drives_rendering_and_physical_demand() {
        let program = fixture_program(StrictMultiplyPath::FiniteNormalExact);
        let rendered = render_fixture(&program);
        assert!(rendered.contains("uint sum = a + b"));
        assert!(rendered.contains("ulong product = ulong(sig_a_scaled) * ulong(sig_b_scaled)"));
        assert!(!rendered.contains("f32_mul(a, b)"));

        let demands = demand_fixture(&program);
        assert!(demands
            .iter()
            .any(|d| { d.primitive == PhysicalPrimitive::U64Multiply && d.count == 1 }));
        assert!(demands
            .iter()
            .any(|d| d.primitive == PhysicalPrimitive::U64Shift && d.count == 2));
        assert!(demands
            .iter()
            .any(|d| d.primitive == PhysicalPrimitive::U32AddSub && d.count >= 4));
    }

    #[test]
    fn current_vocabulary_fails_atomically_with_aggregate_obligations() {
        let failure = construct_current_vocabulary(ClosureEvidence::default()).unwrap_err();
        assert!(failure.obligations.len() > 10);
        assert!(failure.obligations.iter().any(|o| {
            o.kind == ObligationKind::PreserveOperationDiscriminant
                && o.subject == Subject::Scalar(ScalarEmissionFamily::F32AddSub)
        }));
        assert!(failure.obligations.iter().any(|o| {
            o.kind == ObligationKind::SupplyNativeRealizationBounds
                && o.subject == Subject::Mechanism(RepresentativeMechanism::ScheduleAndResidency)
        }));
        assert!(failure.obligations.iter().any(|o| {
            o.kind == ObligationKind::SupplyAddressRelations
                && o.subject == Subject::Mechanism(RepresentativeMechanism::GlobalMemory)
        }));
        assert!(failure.obligations.iter().any(|o| {
            o.kind == ObligationKind::ReplaceOrContractWeakCasProgress
                && o.subject == Subject::Mechanism(RepresentativeMechanism::WeakCasFloatAtomic)
        }));
    }

    #[test]
    fn weak_cas_is_the_fatal_current_semantics_obligation() {
        let evidence = ClosureEvidence {
            native_realization_bounds: true,
            invocation_shape: true,
            path_cohorts: true,
            address_relations: true,
            strict_f32_multiply_path: Some(StrictMultiplyPath::FiniteNormalExact),
            weak_cas_retry_bound: None,
        };
        assert_eq!(
            audit_mechanism(RepresentativeMechanism::WeakCasFloatAtomic, evidence),
            vec![ObligationKind::ReplaceOrContractWeakCasProgress]
        );
    }

    #[test]
    fn closed_program_type_has_no_partial_construction_state() {
        let operation = strict_f32_multiply(StrictMultiplyPath::FiniteNormalExactShiftProduct);
        assert!(operation
            .nodes
            .iter()
            .any(|node| matches!(node, PhysicalNode::U64Multiply { .. })));
    }
}
