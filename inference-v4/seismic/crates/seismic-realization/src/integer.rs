//! Exact scalar integer operations of the retained Cranelift IR. Consumers keep
//! their own policies for unknown values and traps; arithmetic has one owner.
use cranelift_codegen::ir::{InstructionData as Data, Opcode as O, Type, condcodes::IntCC, types};

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
