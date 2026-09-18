//! Exact scalar integer operations of the retained Cranelift IR. Consumers keep
//! their own policies for unknown values and traps; arithmetic has one owner.
use cranelift_codegen::ir::{condcodes::IntCC, types, InstructionData as Data, Opcode as O, Type};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trap {
    DivisionByZero,
    SignedDivisionOverflow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Evaluation {
    Exact(u64),
    /// A valid result is unknown. `may_trap` means the available operands do not
    /// establish that this execution is defined; it is not a guessed result.
    Unknown {
        may_trap: bool,
    },
    Unsupported,
    Trap(Trap),
}

pub fn mask(ty: Type) -> Option<u64> {
    match ty {
        types::I8 => Some(u8::MAX as u64),
        types::I16 => Some(u16::MAX as u64),
        types::I32 => Some(u32::MAX as u64),
        types::I64 => Some(u64::MAX),
        _ => None,
    }
}
pub fn unsigned(value: u64, ty: Type) -> Option<u64> {
    Some(value & mask(ty)?)
}
pub fn signed(value: u64, ty: Type) -> Option<i64> {
    mask(ty)?;
    Some(((value << (64 - ty.bits())) as i64) >> (64 - ty.bits()))
}

/// Operands are supplied by index from the existing instruction. No graph or
/// instruction clone is needed, and immediates remain part of InstructionData.
pub fn evaluate(
    data: &Data,
    output: Type,
    mut operand: impl FnMut(usize) -> Option<(Type, Option<u64>)>,
) -> Evaluation {
    let Some(output_mask) = mask(output) else {
        return Evaluation::Unsupported;
    };
    let exact = |value| Evaluation::Exact(value & output_mask);
    let unknown = Evaluation::Unknown { may_trap: false };
    match *data {
        Data::Ternary {
            opcode: O::Select, ..
        } => {
            let Some((condition_type, condition)) = operand(0) else {
                return Evaluation::Unsupported;
            };
            let Some((left_type, left)) = operand(1) else {
                return Evaluation::Unsupported;
            };
            let Some((right_type, right)) = operand(2) else {
                return Evaluation::Unsupported;
            };
            if condition_type != types::I8 || left_type != output || right_type != output {
                return Evaluation::Unsupported;
            }
            let selected = match condition {
                Some(c) => {
                    if c & 0xff != 0 {
                        left
                    } else {
                        right
                    }
                }
                None if left == right => left,
                None => None,
            };
            return selected.map(exact).unwrap_or(unknown);
        }
        Data::UnaryImm {
            opcode: O::Iconst,
            imm,
        } => return exact(imm.bits() as u64),
        Data::IntCompare {
            opcode: O::Icmp,
            cond,
            ..
        } => {
            let Some((ty, a)) = operand(0) else {
                return Evaluation::Unsupported;
            };
            let Some((bty, b)) = operand(1) else {
                return Evaluation::Unsupported;
            };
            if ty != bty || mask(ty).is_none() || output != types::I8 {
                return Evaluation::Unsupported;
            }
            return match a.zip(b) {
                Some((a, b)) => exact(u64::from(compare(cond, a, b, ty))),
                None => unknown,
            };
        }
        Data::IntCompareImm {
            opcode: O::IcmpImm,
            cond,
            imm,
            ..
        } => {
            let Some((ty, a)) = operand(0) else {
                return Evaluation::Unsupported;
            };
            if mask(ty).is_none() || output != types::I8 {
                return Evaluation::Unsupported;
            }
            return a
                .map(|a| exact(u64::from(compare(cond, a, imm.bits() as u64, ty))))
                .unwrap_or(unknown);
        }
        _ => {}
    }
    let op = data.opcode();
    let admitted = matches!(
        op,
        O::Iadd
            | O::IaddImm
            | O::Isub
            | O::Imul
            | O::ImulImm
            | O::Band
            | O::BandImm
            | O::Bor
            | O::BorImm
            | O::Bxor
            | O::BxorImm
            | O::Ishl
            | O::IshlImm
            | O::Ushr
            | O::UshrImm
            | O::Sshr
            | O::SshrImm
            | O::Udiv
            | O::UdivImm
            | O::Urem
            | O::UremImm
            | O::Sdiv
            | O::SdivImm
            | O::Srem
            | O::SremImm
            | O::Ineg
            | O::Bnot
            | O::Ireduce
            | O::Uextend
            | O::Sextend
    );
    if !admitted {
        return Evaluation::Unsupported;
    }
    let Some((input_type, a)) = operand(0) else {
        return Evaluation::Unsupported;
    };
    let Some(input_mask) = mask(input_type) else {
        return Evaluation::Unsupported;
    };
    let a = a.map(|a| a & input_mask);
    if matches!(op, O::Ireduce | O::Uextend | O::Sextend) {
        if (op == O::Ireduce && output.bits() >= input_type.bits())
            || (op != O::Ireduce && output.bits() <= input_type.bits())
        {
            return Evaluation::Unsupported;
        }
        return match a {
            Some(a) if op == O::Sextend => exact(signed(a, input_type).unwrap() as u64),
            Some(a) => exact(a),
            None => unknown,
        };
    }
    if input_type != output {
        return Evaluation::Unsupported;
    }
    if matches!(op, O::Ineg | O::Bnot) {
        return a
            .map(|a| exact(if op == O::Ineg { a.wrapping_neg() } else { !a }))
            .unwrap_or(unknown);
    }
    let shift = matches!(
        op,
        O::Ishl | O::IshlImm | O::Ushr | O::UshrImm | O::Sshr | O::SshrImm
    );
    let b = if let Data::BinaryImm64 { imm, .. } = *data {
        Some(imm.bits() as u64)
    } else {
        let Some((ty, b)) = operand(1) else {
            return Evaluation::Unsupported;
        };
        let Some(bits) = mask(ty) else {
            return Evaluation::Unsupported;
        };
        if !shift && ty != output {
            return Evaluation::Unsupported;
        }
        b.map(|b| b & bits)
    };
    let b = b.map(|b| if shift { b } else { b & output_mask });
    if matches!(
        op,
        O::Udiv | O::UdivImm | O::Urem | O::UremImm | O::Sdiv | O::SdivImm | O::Srem | O::SremImm
    ) {
        let Some(b) = b else {
            return Evaluation::Unknown { may_trap: true };
        };
        if b == 0 {
            return Evaluation::Trap(Trap::DivisionByZero);
        }
        if matches!(op, O::Sdiv | O::SdivImm) && signed(b, output) == Some(-1) {
            let Some(a) = a else {
                return Evaluation::Unknown { may_trap: true };
            };
            if signed(a, output) == Some(-(1i128 << (output.bits() - 1)) as i64) {
                return Evaluation::Trap(Trap::SignedDivisionOverflow);
            }
        }
        // Unlike division, Cranelift signed remainder cannot overflow.
        if matches!(op, O::Srem | O::SremImm) && signed(b, output) == Some(-1) {
            return exact(0);
        }
    }
    // Absorbing integer operands determine the value without guessing the other
    // operand. Operand evaluation/trapping remains the caller's responsibility.
    if matches!(op, O::Imul | O::ImulImm | O::Band | O::BandImm) && (a == Some(0) || b == Some(0)) {
        return exact(0);
    }
    if matches!(op, O::Bor | O::BorImm) && (a == Some(output_mask) || b == Some(output_mask)) {
        return exact(output_mask);
    }
    if matches!(op, O::Urem | O::UremImm) && b == Some(1) {
        return exact(0);
    }
    let Some((a, b)) = a.zip(b) else {
        return unknown;
    };
    exact(match op {
        O::Iadd | O::IaddImm => a.wrapping_add(b),
        O::Isub => a.wrapping_sub(b),
        O::Imul | O::ImulImm => a.wrapping_mul(b),
        O::Band | O::BandImm => a & b,
        O::Bor | O::BorImm => a | b,
        O::Bxor | O::BxorImm => a ^ b,
        O::Ishl | O::IshlImm => a.wrapping_shl((b % u64::from(output.bits())) as u32),
        O::Ushr | O::UshrImm => a.wrapping_shr((b % u64::from(output.bits())) as u32),
        O::Sshr | O::SshrImm => {
            (signed(a, output).unwrap() >> (b % u64::from(output.bits()))) as u64
        }
        O::Udiv | O::UdivImm => a / b,
        O::Urem | O::UremImm => a % b,
        O::Sdiv | O::SdivImm => (signed(a, output).unwrap() / signed(b, output).unwrap()) as u64,
        O::Srem | O::SremImm => (signed(a, output).unwrap() % signed(b, output).unwrap()) as u64,
        _ => return Evaluation::Unsupported,
    })
}

/// Project the same wrapping operations onto their low bits. These operations
/// commute with reduction modulo 2^bits, so upper operand bits cannot change the
/// result. This is a derived fact about existing instructions, never a substitute
/// execution or an assumption that an arbitrary offset is aligned.
pub fn low_bits(
    data: &Data,
    output: Type,
    bits: u32,
    mut operand: impl FnMut(usize, u32) -> Option<(Type, Option<u64>)>,
) -> Option<u64> {
    if bits == 0 || bits > output.bits() as u32 || mask(output).is_none() {
        return None;
    }
    if !matches!(
        data.opcode(),
        O::Iconst
            | O::Iadd
            | O::IaddImm
            | O::Isub
            | O::Imul
            | O::ImulImm
            | O::Band
            | O::BandImm
            | O::Bor
            | O::BorImm
            | O::Bxor
            | O::BxorImm
            | O::Ineg
            | O::Bnot
            | O::Ireduce
            | O::Uextend
            | O::Sextend
    ) {
        return None;
    }
    let bit_mask=if bits==64 {u64::MAX}else{(1u64<<bits)-1};
    // An immediate is an operand too: project it into the same residue ring.
    let mut projected=data.clone();
    if let Data::BinaryImm64{imm,..}=&mut projected {
        *imm=cranelift_codegen::ir::immediates::Imm64::new((imm.bits() as u64 & bit_mask) as i64);
    }
    let result = evaluate(&projected, output, |index| operand(index, bits));
    let Evaluation::Exact(value) = result else {return None;};
    Some(value & bit_mask)
}

fn compare(cond: IntCC, a: u64, b: u64, ty: Type) -> bool {
    let a = unsigned(a, ty).unwrap();
    let b = unsigned(b, ty).unwrap();
    match cond {
        IntCC::Equal => a == b,
        IntCC::NotEqual => a != b,
        IntCC::SignedLessThan => signed(a, ty) < signed(b, ty),
        IntCC::SignedLessThanOrEqual => signed(a, ty) <= signed(b, ty),
        IntCC::SignedGreaterThan => signed(a, ty) > signed(b, ty),
        IntCC::SignedGreaterThanOrEqual => signed(a, ty) >= signed(b, ty),
        IntCC::UnsignedLessThan => a < b,
        IntCC::UnsignedLessThanOrEqual => a <= b,
        IntCC::UnsignedGreaterThan => a > b,
        IntCC::UnsignedGreaterThanOrEqual => a >= b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cranelift_codegen::ir::Value;

    fn binary(ty: Type, opcode: O, a: Option<u64>, b: Option<u64>) -> Evaluation {
        let data = Data::Binary {
            opcode,
            args: [Value::from_u32(0), Value::from_u32(1)],
        };
        evaluate(&data, ty, |index| {
            [a, b].get(index).map(|&value| (ty, value))
        })
    }

    #[test]
    fn integer_select_and_absorbing_values_preserve_partial_information() {
        let data = Data::Ternary {
            opcode: O::Select,
            args: [Value::from_u32(0), Value::from_u32(1), Value::from_u32(2)],
        };
        for (condition, left, right, expected) in [
            (Some(1), Some(7), None, Evaluation::Exact(7)),
            (Some(0), None, Some(9), Evaluation::Exact(9)),
            (None, Some(5), Some(5), Evaluation::Exact(5)),
            (
                None,
                Some(5),
                Some(9),
                Evaluation::Unknown { may_trap: false },
            ),
        ] {
            assert_eq!(
                evaluate(&data, types::I32, |i| Some((
                    if i == 0 { types::I8 } else { types::I32 },
                    [condition, left, right][i]
                ))),
                expected
            );
        }
        for op in [O::Imul, O::Band] {
            assert_eq!(binary(types::I32, op, None, Some(0)), Evaluation::Exact(0));
        }
        assert_eq!(
            binary(types::I32, O::Urem, None, Some(1)),
            Evaluation::Exact(0)
        );
    }

    #[test]
    fn low_bit_projection_is_independent_of_every_discarded_operand_bit() {
        for bits in 1..=8 {
            let mask = (1u64 << bits) - 1;
            for op in [O::Iadd, O::Isub, O::Imul, O::Band, O::Bor, O::Bxor] {
                let data = Data::Binary {
                    opcode: op,
                    args: [Value::from_u32(0), Value::from_u32(1)],
                };
                for a in 0..=255u64 {
                    for b in [0, 1, 63, 128, 255] {
                        let projected = low_bits(&data, types::I8, bits, |i, _| {
                            Some((types::I8, Some([a, b][i] & mask)))
                        })
                        .unwrap();
                        let Evaluation::Exact(actual) = binary(types::I8, op, Some(a), Some(b))
                        else {
                            panic!()
                        };
                        assert_eq!(projected, actual & mask);
                    }
                }
            }
        }
        let product = Data::Binary {
            opcode: O::Imul,
            args: [Value::from_u32(0), Value::from_u32(1)],
        };
        assert_eq!(
            low_bits(&product, types::I64, 6, |i, _| Some((
                types::I64,
                if i == 0 { None } else { Some(0) }
            ))),
            Some(0)
        );
        let aligned=Data::BinaryImm64{opcode:O::ImulImm,arg:Value::from_u32(0),imm:cranelift_codegen::ir::immediates::Imm64::new(256)};
        assert_eq!(low_bits(&aligned,types::I64,6,|_,_|Some((types::I64,None))),Some(0));
        let remainder = Data::Binary {
            opcode: O::Urem,
            args: [Value::from_u32(0), Value::from_u32(1)],
        };
        assert_eq!(
            low_bits(&remainder, types::I64, 6, |_, _| Some((
                types::I64,
                Some(0)
            ))),
            None
        );
    }

    #[test]
    fn wrapping_is_at_the_instruction_width() {
        for a in 0..=u8::MAX {
            for b in [0, 1, 127, 128, 255] {
                for (op, expected) in [
                    (O::Iadd, a.wrapping_add(b)),
                    (O::Isub, a.wrapping_sub(b)),
                    (O::Imul, a.wrapping_mul(b)),
                ] {
                    assert_eq!(
                        binary(types::I8, op, Some(u64::from(a)), Some(u64::from(b))),
                        Evaluation::Exact(u64::from(expected))
                    );
                }
            }
        }
        assert_eq!(
            binary(types::I8, O::Iadd, Some(0x180), Some(0x80)),
            Evaluation::Exact(0)
        );
        assert_eq!(
            binary(types::I64, O::Iadd, Some(u64::MAX), Some(1)),
            Evaluation::Exact(0)
        );
        assert_eq!(
            binary(types::I64, O::Iadd, Some(i64::MAX as u64), Some(1)),
            Evaluation::Exact(i64::MIN as u64)
        );
        assert_eq!(
            binary(types::I64, O::Imul, Some(u64::MAX), Some(2)),
            Evaluation::Exact(u64::MAX - 1)
        );
        assert_eq!(
            binary(types::I8, O::Ishl, Some(3), Some(8)),
            Evaluation::Exact(3)
        );
        assert_eq!(
            binary(types::I64, O::Ushr, Some(8), Some(65)),
            Evaluation::Exact(4)
        );
    }

    #[test]
    fn division_traps_and_remainder_does_not_overflow() {
        for (ty, min, minus_one) in [
            (types::I8, 128, 255),
            (types::I64, i64::MIN as u64, u64::MAX),
        ] {
            assert_eq!(
                binary(ty, O::Sdiv, Some(min), Some(minus_one)),
                Evaluation::Trap(Trap::SignedDivisionOverflow)
            );
            assert_eq!(
                binary(ty, O::Srem, Some(min), Some(minus_one)),
                Evaluation::Exact(0)
            );
            assert_eq!(
                binary(ty, O::Srem, None, Some(minus_one)),
                Evaluation::Exact(0)
            );
            assert_eq!(
                binary(ty, O::Sdiv, None, Some(minus_one)),
                Evaluation::Unknown { may_trap: true }
            );
            for op in [O::Sdiv, O::Srem, O::Udiv, O::Urem] {
                assert_eq!(
                    binary(ty, op, None, Some(0)),
                    Evaluation::Trap(Trap::DivisionByZero)
                );
                assert_eq!(
                    binary(ty, op, Some(7), None),
                    Evaluation::Unknown { may_trap: true }
                );
                assert_eq!(
                    binary(ty, op, None, Some(2)),
                    Evaluation::Unknown { may_trap: false }
                );
            }
        }
        assert_eq!(
            binary(types::I8, O::Udiv, Some(0x1ff), Some(2)),
            Evaluation::Exact(127)
        );
    }

    #[test]
    fn comparisons_and_conversions_use_declared_signedness() {
        let argument = Value::from_u32(0);
        for (cond, expected) in [
            (IntCC::Equal, 0),
            (IntCC::SignedLessThan, 1),
            (IntCC::UnsignedLessThan, 0),
        ] {
            let data = Data::IntCompare {
                opcode: O::Icmp,
                cond,
                args: [argument, Value::from_u32(1)],
            };
            assert_eq!(
                evaluate(&data, types::I8, |i| Some((types::I8, Some([0x1ff, 1][i])))),
                Evaluation::Exact(expected)
            );
        }
        let extend = Data::Unary {
            opcode: O::Sextend,
            arg: argument,
        };
        assert_eq!(
            evaluate(&extend, types::I64, |_| Some((types::I8, Some(255)))),
            Evaluation::Exact(u64::MAX)
        );
        assert_eq!(
            binary(types::F32, O::Iadd, Some(0), Some(0)),
            Evaluation::Unsupported
        );
    }
}
