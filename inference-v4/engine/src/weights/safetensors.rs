//! Header-only interpretation of the stored dtypes admitted by V3's MLX loader.
use super::Error;
use seismic_lang::types::DType;
use serde::{
    de::{self, MapAccess, Visitor},
    Deserialize, Deserializer,
};
use std::{
    collections::HashSet,
    fmt,
    io::{Read, Seek, SeekFrom},
};

pub const HEADER_LIMIT: u64 = 64 * 1024 * 1024;
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    /// Absolute byte offset in the source file.
    pub offset: u64,
    pub nbytes: u64,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Directory {
    pub data_offset: u64,
    pub tensors: Vec<Tensor>,
    pub metadata: Option<serde_json::Value>,
}
#[derive(Deserialize)]
struct Entry {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}
struct Entries(Vec<(String, serde_json::Value)>);
impl<'de> Deserialize<'de> for Entries {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct EntriesVisitor;
        impl<'de> Visitor<'de> for EntriesVisitor {
            type Value = Entries;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a Safetensors directory object")
            }
            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Entries, M::Error> {
                let mut names = HashSet::new();
                let mut entries = Vec::new();
                while let Some((name, value)) = map.next_entry::<String, serde_json::Value>()? {
                    if !names.insert(name.clone()) {
                        return Err(de::Error::custom(format!(
                            "duplicate Safetensors entry {name:?}"
                        )));
                    }
                    entries.push((name, value));
                }
                Ok(Entries(entries))
            }
        }
        d.deserialize_map(EntriesVisitor)
    }
}
fn invalid(s: impl Into<String>) -> Error {
    Error::Invalid(s.into())
}
pub fn read_directory<R: Read + Seek>(source: &mut R) -> Result<Directory, Error> {
    let size = source.seek(SeekFrom::End(0))?;
    source.seek(SeekFrom::Start(0))?;
    let mut raw = [0; 8];
    source.read_exact(&mut raw)?;
    let length = u64::from_le_bytes(raw);
    if length > HEADER_LIMIT || length > size.saturating_sub(8) {
        return Err(invalid("invalid Safetensors header extent"));
    }
    let mut header = vec![
        0;
        usize::try_from(length)
            .map_err(|_| invalid("Safetensors header exceeds address range"))?
    ];
    source.read_exact(&mut header)?;
    let Entries(entries) = serde_json::from_slice(&header)
        .map_err(|e| invalid(format!("invalid Safetensors header: {e}")))?;
    let data_offset = 8 + length;
    let mut tensors = Vec::new();
    let mut metadata = None;
    for (name, value) in entries {
        if name == "__metadata__" {
            metadata = Some(value);
            continue;
        }
        let entry: Entry = serde_json::from_value(value)
            .map_err(|e| invalid(format!("invalid Safetensors tensor {name:?}: {e}")))?;
        let dtype = match entry.dtype.as_str() {
            "U32" => DType::U32,
            "BF16" => DType::BF16,
            "F32" => DType::F32,
            _ => {
                return Err(invalid(format!(
                    "unsupported Safetensors dtype {:?}",
                    entry.dtype
                )))
            }
        };
        if entry.shape.contains(&0) {
            return Err(invalid(format!("invalid Safetensors shape {name:?}")));
        }
        let nbytes = entry
            .shape
            .iter()
            .try_fold(u64::from(dtype.bytes()), |n, d| n.checked_mul(*d))
            .ok_or_else(|| invalid("Safetensors shape overflows"))?;
        let [start, end] = entry.data_offsets;
        if end.checked_sub(start) != Some(nbytes) || end > size - data_offset {
            return Err(invalid(format!("invalid Safetensors range {name:?}")));
        }
        tensors.push(Tensor {
            name,
            dtype,
            shape: entry.shape,
            offset: data_offset + start,
            nbytes,
        });
    }
    let mut ranges = tensors.iter().collect::<Vec<_>>();
    ranges.sort_by_key(|t| t.offset);
    if ranges
        .windows(2)
        .any(|w| w[1].offset < w[0].offset + w[0].nbytes)
    {
        return Err(invalid("overlapping Safetensors ranges"));
    }
    Ok(Directory {
        data_offset,
        tensors,
        metadata,
    })
}
