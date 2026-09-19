//! Uniform control and address interpretation over declared invocation domains.
//! Values remain affine over the original inputs; no representative invocation
//! is substituted for a range and no instruction is eliminated by propagation.
use super::*;
use cranelift_codegen::ir::Opcode as O;

pub(super) fn evaluate(instruction: &Instruction, inputs: &[&Datum]) -> Result<Option<Datum>, DerivationError> {
    let output = instruction.outputs.first().map(|(_, ty)| *ty).unwrap_or(types::I64);
    if seismic_realization::integer::mask(output).is_none() { return Ok(None); }
    let input_type = instruction.inputs.first().map(|v| v.ty).unwrap_or(output);
    let op = instruction.opcode;
    let finish = |value: Option<Affine>| {
        value.and_then(|v| v.interpreted(output.bits(), false)).map_or(Datum::Unknown, |value| {
            value.exact().map_or_else(|| Datum::Integer(value.clone()), |v| Datum::Bits(v as u64))
        })
    };
    let a = |signed| inputs.first().and_then(|v| v.integer(input_type, signed));
    let b = |signed| {
        match instruction.encoding {
            Data::BinaryImm64 { imm, .. } | Data::IntCompareImm { imm, .. } =>
                Datum::Bits(imm.bits() as u64).integer(input_type, signed),
            _ => inputs.get(1).and_then(|v| v.integer(instruction.inputs.get(1)?.ty, signed)),
        }
    };
    match instruction.encoding {
        Data::IntCompare { cond, .. } | Data::IntCompareImm { cond, .. } => {
            let signed = matches!(cond, IntCC::SignedLessThan | IntCC::SignedLessThanOrEqual | IntCC::SignedGreaterThan | IntCC::SignedGreaterThanOrEqual);
            let Some((lo, hi)) = a(signed).zip(b(signed)).and_then(|(a,b)| a.sub(&b)?.bounds()) else {
                return Ok(Some(Datum::Unknown));
            };
            let truth = match cond {
                IntCC::Equal => if lo == 0 && hi == 0 { Some(true) } else if hi < 0 || lo > 0 { Some(false) } else { None },
                IntCC::NotEqual => if lo == 0 && hi == 0 { Some(false) } else if hi < 0 || lo > 0 { Some(true) } else { None },
                IntCC::SignedLessThan | IntCC::UnsignedLessThan => if hi < 0 { Some(true) } else if lo >= 0 { Some(false) } else { None },
                IntCC::SignedLessThanOrEqual | IntCC::UnsignedLessThanOrEqual => if hi <= 0 { Some(true) } else if lo > 0 { Some(false) } else { None },
                IntCC::SignedGreaterThan | IntCC::UnsignedGreaterThan => if lo > 0 { Some(true) } else if hi <= 0 { Some(false) } else { None },
                IntCC::SignedGreaterThanOrEqual | IntCC::UnsignedGreaterThanOrEqual => if lo >= 0 { Some(true) } else if hi < 0 { Some(false) } else { None },
            };
            return Ok(Some(truth.map_or(Datum::Unknown, |v| Datum::Bits(u64::from(v)))));
        }
        _ => {}
    }
    let value = match op {
        O::Iadd | O::IaddImm => a(false).zip(b(false)).and_then(|(a,b)| a.add(&b)),
        O::Isub => a(false).zip(b(false)).and_then(|(a,b)| a.sub(&b)),
        O::Imul | O::ImulImm => a(false).zip(b(false)).and_then(|(a,b)| {
            if let Some(c) = a.exact() { b.scale(c) } else { a.scale(b.exact()?) }
        }),
        O::Ineg => a(false).and_then(|a| a.scale(-1)),
        O::Bnot => a(false).and_then(|a| a.scale(-1)?.add(&Affine::constant(-1))),
        O::Ireduce | O::Uextend => a(false),
        O::Sextend => a(true),
        O::Ishl | O::IshlImm | O::Ushr | O::UshrImm | O::Sshr | O::SshrImm => {
            let shift = b(false).and_then(|b| b.exact()).map(|n| n.rem_euclid(i128::from(input_type.bits())) as u32);
            let signed = matches!(op, O::Sshr | O::SshrImm);
            a(signed).zip(shift).and_then(|(a, shift)| {
                if matches!(op, O::Ishl | O::IshlImm) { a.scale(1i128 << shift) }
                else { a.quotient(1u64 << shift) }
            })
        }
        O::Udiv | O::UdivImm | O::Urem | O::UremImm | O::Sdiv | O::SdivImm | O::Srem | O::SremImm => {
            let signed = matches!(op, O::Sdiv | O::SdivImm | O::Srem | O::SremImm);
            let dividend = a(signed);
            let divisor = b(signed);
            let Some((lo, hi)) = divisor.as_ref().and_then(Affine::bounds) else {
                return Err(DerivationError::Unsupported("scalar workload domain does not establish a defined integer divisor".into()));
            };
            if lo <= 0 && hi >= 0 {
                return Err(DerivationError::Unsupported("scalar workload domain includes a possibly zero integer divisor".into()));
            }
            if matches!(op, O::Sdiv | O::SdivImm) && lo <= -1 && hi >= -1 {
                let minimum = -(1i128 << (output.bits()-1));
                if dividend.as_ref().and_then(Affine::bounds).is_none_or(|(lo,hi)| lo <= minimum && hi >= minimum) {
                    return Err(DerivationError::Unsupported("scalar workload domain does not exclude signed division overflow".into()));
                }
            }
            let result = dividend.zip(divisor).and_then(|(a,b)| {
                let divisor = b.exact()?;
                let remainder = matches!(op, O::Urem | O::UremImm | O::Srem | O::SremImm);
                let magnitude = u64::try_from(divisor.unsigned_abs()).ok()?;
                let (lo, hi) = a.bounds()?;
                let quotient = if !signed || lo >= 0 { a.quotient(magnitude)? }
                    else if hi <= 0 { a.scale(-1)?.quotient(magnitude)?.scale(-1)? }
                    else if magnitude == 1 { a.clone() }
                    else { return None; };
                let quotient = if divisor < 0 { quotient.scale(-1)? } else { quotient };
                if remainder { a.sub(&quotient.scale(divisor)?) } else { Some(quotient) }
            });
            return Ok(Some(finish(result)));
        }
        // Bit masks with a contiguous low field are quotient/remainder
        // operations and preserve the original independent coordinates.
        O::Band | O::BandImm => a(false).zip(b(false)).and_then(|(a,b)| {
            let (a, mask) = if let Some(mask) = b.exact() { (a, mask) } else { (b, a.exact()?) };
            let divisor = u64::try_from(mask.checked_add(1)?).ok()?;
            divisor.is_power_of_two().then(|| a.remainder(divisor)).flatten()
        }),
        O::Bxor | O::BxorImm if a(false).is_some() && a(false) == b(false) => Some(Affine::constant(0)),
        O::Bor | O::BorImm | O::Bxor | O::BxorImm => None,
        _ => return Ok(None),
    };
    Ok(Some(finish(value)))
}
