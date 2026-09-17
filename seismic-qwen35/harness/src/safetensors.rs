//! Reading tensors from a safetensors file by name.

use crate::json::Json;
use std::collections::BTreeMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub start: u64,
    pub end: u64,
}

pub struct SafeTensors {
    file: File,
    base: u64,
    pub tensors: BTreeMap<String, TensorInfo>,
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<SafeTensors, String> {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut len = [0u8; 8];
        file.read_exact_at(&mut len, 0).map_err(|e| e.to_string())?;
        let n = u64::from_le_bytes(len);
        let mut header = vec![0u8; n as usize];
        file.read_exact_at(&mut header, 8).map_err(|e| e.to_string())?;
        let text = std::str::from_utf8(&header).map_err(|_| "header is not UTF-8")?;
        let json = Json::parse(text)?;
        let mut tensors = BTreeMap::new();
        for (name, v) in json.as_object()? {
            if name == "__metadata__" {
                continue;
            }
            let offsets = v.get("data_offsets")?.as_array()?;
            tensors.insert(
                name.clone(),
                TensorInfo {
                    dtype: v.get("dtype")?.as_str()?.to_string(),
                    shape: v.get("shape")?.as_array()?.iter().map(|d| d.as_i64().map(|x| x as usize)).collect::<Result<_, _>>()?,
                    start: offsets[0].as_i64()? as u64,
                    end: offsets[1].as_i64()? as u64,
                },
            );
        }
        Ok(SafeTensors { file, base: 8 + n, tensors })
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo, String> {
        self.tensors.get(name).ok_or_else(|| format!("no tensor `{name}`"))
    }

    pub fn read(&self, name: &str) -> Result<Vec<u8>, String> {
        let t = self.info(name)?;
        let mut out = vec![0u8; (t.end - t.start) as usize];
        self.file.read_exact_at(&mut out, self.base + t.start).map_err(|e| format!("{name}: {e}"))?;
        Ok(out)
    }
}
