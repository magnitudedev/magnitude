//! Typed storage-plane geometry shared by backend binding paths.
use seismic_lang::{repr, types::Elem};
pub fn plane_byte_offset(elem: &Elem, plane: &str, element_offset: i64) -> Result<usize, String> {
    let offset = usize::try_from(element_offset).map_err(|_| "negative tensor view offset")?;
    let (group, bytes) = match elem {
        Elem::Dtype(dtype) if plane.is_empty() => (1, dtype.bytes() as usize),
        Elem::Dtype(_) => return Err("dense storage has no named packed plane".into()),
        Elem::Repr(name) => {
            let rep =
                repr::lookup(name).ok_or_else(|| format!("unknown representation `{name}`"))?;
            match plane {
                "words" => (rep.codes_per_word() as usize, 4),
                "scale" => (rep.group as usize, rep.coefficient.bytes() as usize),
                "bias" if rep.has_bias => (rep.group as usize, rep.coefficient.bytes() as usize),
                _ => return Err(format!("representation `{name}` has no plane `{plane}`")),
            }
        }
        Elem::Param(name) => return Err(format!("unbound storage element `{name}`")),
    };
    if !offset.is_multiple_of(group) {
        return Err(format!(
            "view offset {offset} is not aligned to {group}-element storage groups"
        ));
    }
    (offset / group)
        .checked_mul(bytes)
        .ok_or_else(|| "storage plane offset overflow".into())
}
#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::types::DType;
    #[test]
    fn planes_preserve_packet_geometry_and_reject_partial_groups() {
        let packed = Elem::Repr("q4g64".into());
        assert_eq!(plane_byte_offset(&packed, "words", 128).unwrap(), 64);
        assert_eq!(plane_byte_offset(&packed, "scale", 128).unwrap(), 4);
        assert_eq!(plane_byte_offset(&packed, "bias", 128).unwrap(), 4);
        assert!(plane_byte_offset(&packed, "scale", 8).is_err());
        assert!(plane_byte_offset(&packed, "words", -1).is_err());
        assert!(plane_byte_offset(&packed, "unknown", 0).is_err());
        assert_eq!(
            plane_byte_offset(&Elem::Dtype(DType::F32), "", 3).unwrap(),
            12
        );
        assert!(plane_byte_offset(&Elem::Dtype(DType::F32), "words", 0).is_err());
    }
}
