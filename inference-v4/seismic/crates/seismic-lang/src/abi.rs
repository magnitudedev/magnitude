//! Derived scalar invocation layouts. No per-kernel handwritten byte packing.
use crate::{
    numeric::{bf16_round, f16_bits},
    types::DType,
};
/// One scalar's semantic contract, shared by every target ABI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalarParameter {
    pub name: String,
    pub dtype: DType,
    /// Exclusive upper bound for a declared index. Zero admits no values.
    pub index_bound: Option<u64>,
}
impl ScalarParameter {
    pub fn plain(name: impl Into<String>, dtype: DType) -> Self {
        Self {
            name: name.into(),
            dtype,
            index_bound: None,
        }
    }
    pub fn from_lowered(
        lowered: &crate::lower::Lowered,
        name: &str,
        dtype: DType,
    ) -> Result<Self, String> {
        let index_bound = lowered
            .index_params
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, bound)| {
                bound
                    .as_constant()
                    .and_then(|n| u64::try_from(n).ok())
                    .ok_or_else(|| format!("unresolved or negative index bound for `{name}`"))
            })
            .transpose()?;
        if index_bound.is_some() && dtype != DType::I32 {
            return Err("index parameter must have i32 storage".into());
        }
        Ok(Self {
            name: name.into(),
            dtype,
            index_bound,
        })
    }
}
#[derive(Clone, Debug)]
pub struct ScalarField {
    pub parameter: ScalarParameter,
    pub offset: usize,
}
#[derive(Clone, Debug)]
pub struct ScalarLayout {
    pub fields: Vec<ScalarField>,
    pub bytes: usize,
}
impl ScalarLayout {
    /// Natural C/MSL scalar struct alignment, including trailing padding.
    pub fn natural(parameters: &[ScalarParameter]) -> Result<Self, String> {
        Self::layout(parameters, None)
    }
    /// Every scalar occupies a zero-padded eight-byte slot in the shared scalar ABI.
    pub fn words(parameters: &[ScalarParameter]) -> Result<Self, String> {
        Self::layout(parameters, Some(8))
    }
    fn layout(parameters: &[ScalarParameter], slot: Option<usize>) -> Result<Self, String> {
        let mut fields = Vec::new();
        let mut bytes = 0usize;
        let mut max_alignment = 1;
        for parameter in parameters {
            let dtype = parameter.dtype;
            if parameter.index_bound.is_some() && dtype != DType::I32 {
                return Err("index parameter must have i32 storage".into());
            }
            let alignment = slot.unwrap_or(dtype.bytes() as usize);
            max_alignment = max_alignment.max(alignment);
            bytes = align(bytes, alignment)?;
            fields.push(ScalarField {
                parameter: parameter.clone(),
                offset: bytes,
            });
            bytes = bytes
                .checked_add(slot.unwrap_or(dtype.bytes() as usize))
                .ok_or("scalar layout overflow")?;
        }
        Ok(Self {
            fields,
            bytes: align(bytes, max_alignment)?,
        })
    }
    /// Raw-byte callers receive the same domain checks as typed callers.
    pub fn validate_bytes(&self, bytes: &[u8]) -> Result<(), String> {
        if bytes.len() != self.bytes {
            return Err("scalar binding size mismatch".into());
        }
        for field in &self.fields {
            let width = field.parameter.dtype.bytes() as usize;
            let data = bytes
                .get(
                    field.offset
                        ..field
                            .offset
                            .checked_add(width)
                            .ok_or("scalar offset overflow")?,
                )
                .ok_or("scalar field exceeds layout")?;
            if field.parameter.dtype == DType::Bool && data[0] > 1 {
                return Err(format!("invalid boolean `{}`", field.parameter.name));
            }
            if let Some(bound) = field.parameter.index_bound {
                if field.parameter.dtype != DType::I32 {
                    return Err("index parameter must have i32 storage".into());
                }
                let value =
                    i32::from_le_bytes(data.try_into().map_err(|_| "invalid index storage width")?);
                if value < 0 || value as u64 >= bound {
                    return Err(format!(
                        "index `{}` outside its declared domain",
                        field.parameter.name
                    ));
                }
            }
        }
        Ok(())
    }
    pub fn encode(&self, values: &[f64]) -> Result<Vec<u8>, String> {
        if values.len() != self.fields.len() {
            return Err("scalar binding count mismatch".into());
        }
        let mut out = vec![0u8; self.bytes];
        for (field, value) in self.fields.iter().zip(values) {
            if field
                .parameter
                .index_bound
                .is_some_and(|bound| !value.is_finite() || *value < 0.0 || *value >= bound as f64)
            {
                return Err(format!(
                    "index `{}` outside its declared domain",
                    field.parameter.name
                ));
            }
            let bits = match field.parameter.dtype {
                DType::F32 => u64::from((*value as f32).to_bits()),
                DType::BF16 => u64::from(bf16_round(*value as f32).to_bits() >> 16),
                DType::F16 => u64::from(f16_bits(*value as f32)),
                DType::Bool if *value == 0.0 || *value == 1.0 => *value as u64,
                DType::I32
                    if value.is_finite()
                        && value.fract() == 0.0
                        && *value >= i32::MIN as f64
                        && *value <= i32::MAX as f64 =>
                {
                    u64::from(*value as i32 as u32)
                }
                DType::U32
                    if value.is_finite()
                        && value.fract() == 0.0
                        && *value >= 0.0
                        && *value <= u32::MAX as f64 =>
                {
                    u64::from(*value as u32)
                }
                _ => {
                    return Err(format!(
                        "invalid scalar `{}` for {}",
                        field.parameter.name,
                        field.parameter.dtype.name()
                    ))
                }
            };
            let width = field.parameter.dtype.bytes() as usize;
            let end = field
                .offset
                .checked_add(width)
                .ok_or("scalar offset overflow")?;
            out.get_mut(field.offset..end)
                .ok_or("scalar field exceeds layout")?
                .copy_from_slice(&bits.to_le_bytes()[..width]);
        }
        Ok(out)
    }
}
fn align(n: usize, a: usize) -> Result<usize, String> {
    n.checked_add(a - 1)
        .map(|n| n & !(a - 1))
        .ok_or_else(|| "scalar alignment overflow".into())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mixed_width_scalar_layout_and_validation() {
        let schema = vec![
            ScalarParameter::plain("flag", DType::Bool),
            ScalarParameter::plain("x", DType::F32),
            ScalarParameter::plain("h", DType::F16),
            ScalarParameter::plain("b", DType::BF16),
            ScalarParameter::plain("end", DType::Bool),
        ];
        let layout = ScalarLayout::natural(&schema).unwrap();
        assert_eq!(
            layout.fields.iter().map(|f| f.offset).collect::<Vec<_>>(),
            [0, 4, 8, 10, 12]
        );
        assert_eq!(layout.bytes, 16);
        let encoded = layout.encode(&[1.0, 2.0, 3.0, 4.0, 0.0]).unwrap();
        assert_eq!(encoded[0], 1);
        assert_eq!(&encoded[4..8], &2.0f32.to_le_bytes());
        assert_eq!(&encoded[8..10], &0x4200u16.to_le_bytes());
        assert_eq!(&encoded[10..12], &0x4080u16.to_le_bytes());
        assert!(layout.encode(&[2.0, 2.0, 3.0, 4.0, 0.0]).is_err());
        assert_eq!(ScalarLayout::words(&schema).unwrap().bytes, 40);
    }
    #[test]
    fn declared_index_bounds_survive_lowering_and_raw_abi() {
        use crate::{
            program::{compile, SourceFile},
            Scope,
        };
        let program=compile(&[SourceFile{path:"bounded.seismic.portable".into(),scope:Scope::Portable,text:"fn bounded[N](pos: index[N], out: tensor[1] i32):\n  t = tile[1] i32\n  for i in owned(t): t[i] = pos\n  store(t,out)\n".into()}],&[]).unwrap();
        let lowered =
            crate::lower::lower(&program, "bounded", "cpu", &[("N".into(), 4)].into()).unwrap();
        let parameter = ScalarParameter::from_lowered(&lowered, "pos", DType::I32).unwrap();
        assert_eq!(parameter.index_bound, Some(4));
        for layout in [
            ScalarLayout::natural(&[parameter.clone()]).unwrap(),
            ScalarLayout::words(&[parameter]).unwrap(),
        ] {
            for n in [0.0, 3.0] {
                let bytes = layout.encode(&[n]).unwrap();
                layout.validate_bytes(&bytes).unwrap();
            }
            for n in [-1.0, 4.0, 0.5, f64::NAN, f64::INFINITY] {
                assert!(layout.encode(&[n]).is_err());
            }
            for n in [-1i32, 4] {
                let mut bytes = vec![0; layout.bytes];
                bytes[..4].copy_from_slice(&n.to_le_bytes());
                assert!(layout.validate_bytes(&bytes).is_err());
            }
        }
    }
}
