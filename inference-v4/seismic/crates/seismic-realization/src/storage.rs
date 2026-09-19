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
            let descriptor = rep
                .plane(plane)
                .ok_or_else(|| format!("representation `{name}` has no plane `{plane}`"))?;
            return descriptor
                .byte_offset(offset as u64)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| {
                    format!("view offset {offset} does not align to plane `{plane}` storage")
                });
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

/// Entry storage ABI derived solely from specialized parameter types. This is
/// shared by invocation binding and compiler ownership admission before choosing
/// implementations; unused arguments keep their ABI positions.
pub fn parameters(
    function: &seismic_lang::exec::lowered_ir::LoweredIr,
) -> Result<
    (
        Vec<crate::BufferSpec>,
        Vec<seismic_lang::abi::ScalarParameter>,
    ),
    String,
> {
    parameter_types(&function.params, &function.index_params)
}
pub fn parameter_types(
    params: &[(String, seismic_lang::exec::types::Ty)],
    index_params: &[(String, seismic_lang::sym::Sym)],
) -> Result<
    (
        Vec<crate::BufferSpec>,
        Vec<seismic_lang::abi::ScalarParameter>,
    ),
    String,
> {
    use seismic_lang::{abi::ScalarParameter, exec::types::Ty};
    let mut buffers = Vec::new();
    let mut scalars = Vec::new();
    for (name, ty) in params {
        match ty {
            Ty::Tensor(t) => {
                let count = t.shape.iter().try_fold(1u64, |n, x| {
                    x.as_constant()
                        .and_then(|v| u64::try_from(v).ok())
                        .and_then(|v| n.checked_mul(v))
                        .ok_or("unresolved or overflowing entry storage extent")
                })?;
                match &t.elem {
                    Elem::Dtype(d) => buffers.push(crate::BufferSpec {
                        parameter: name.clone(),
                        plane: String::new(),
                        bytes: count
                            .checked_mul(u64::from(d.bytes()))
                            .and_then(|v| usize::try_from(v).ok())
                            .ok_or("entry storage overflow")?,
                        alignment: d.bytes() as usize,
                    }),
                    Elem::Repr(r) => {
                        let r = repr::lookup(r).ok_or("unknown entry representation")?;
                        for plane in r.planes() {
                            buffers.push(crate::BufferSpec {
                                parameter: name.clone(),
                                plane: plane.name.into(),
                                bytes: plane
                                    .storage_elements(count)
                                    .and_then(|v| v.checked_mul(u64::from(plane.dtype().bytes())))
                                    .and_then(|v| usize::try_from(v).ok())
                                    .ok_or("entry representation extent overflow")?,
                                alignment: plane.dtype().bytes() as usize,
                            });
                        }
                    }
                    Elem::Param(_) => return Err("unbound entry representation".into()),
                }
            }
            Ty::Scalar(d) => scalars.push(ScalarParameter {
                name: name.clone(),
                dtype: *d,
                index_bound: index_params
                    .iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, s)| {
                        s.as_constant()
                            .and_then(|n| u64::try_from(n).ok())
                            .ok_or("unresolved index bound")
                    })
                    .transpose()?,
            }),
            _ => return Err("entry parameters must be tensors or scalars".into()),
        }
    }
    Ok((buffers, scalars))
}
