//! Vulkan's program toolchain.
//!
//! Compiling a program compiles its source once per entry (the entry is
//! selected by `SEISMIC_KERNEL`) to sealed, validated SPIR-V, then binds
//! `fma` for the device and creates each entry's pipeline with its constants.
//! It keeps each entry's SPIR-V, with what creating the program's pipelines
//! left in a pipeline cache: a kept module is used when it validates, a
//! program whose modules are not all kept is compiled again, and a driver
//! given the kept pipelines compiles nothing.

use crate::device::Device;
use crate::formation::{self, DirectModule};
use seismic_native_target::{
    NativeCompilationError, ProgramCache, ProgramSource, Toolchain, ToolchainIdentity,
};

pub struct VulkanToolchain {
    device: Device,
    identity: ToolchainIdentity,
}

impl VulkanToolchain {
    pub fn new(device: Device) -> Self {
        let identity = ToolchainIdentity {
            namespace: "vulkan",
            material: formation::formation(&device),
        };
        Self { device, identity }
    }

    /// `source`'s modules from kept bytes, if they hold every entry and each
    /// validates.
    fn load(&self, source: &ProgramSource, kept: &[u8]) -> Option<Vec<Vec<u32>>> {
        let modules = unpack(kept)?;
        source
            .entries
            .iter()
            .map(|entry| {
                modules
                    .iter()
                    .find(|(symbol, _)| *symbol == entry.symbol)
                    .map(|(_, words)| words.clone())
                    .filter(|words| formation::validate(words).is_ok())
            })
            .collect()
    }
}

/// The kept entry that holds a program's pipeline cache data: no symbol of
/// a source, its words the data's length and then the data.
const PIPELINES: &str = "\0pipelines";

fn pipelines_words(data: &[u8]) -> Vec<u32> {
    let mut words = vec![u32::try_from(data.len()).expect("pipeline cache data fits u32")];
    words.extend(data.chunks(4).map(|chunk| {
        let mut word = [0; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        u32::from_le_bytes(word)
    }));
    words
}

fn pipelines_data(words: &[u32]) -> Vec<u8> {
    let Some((length, rest)) = words.split_first() else {
        return Vec::new();
    };
    let mut data = rest.iter().flat_map(|word| word.to_le_bytes()).collect::<Vec<_>>();
    data.truncate(*length as usize);
    data
}

/// Each module's symbol and words, length-prefixed, little-endian.
fn pack(modules: &[(&str, Vec<u32>)]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for (symbol, words) in modules {
        bytes.extend_from_slice(&(symbol.len() as u32).to_le_bytes());
        bytes.extend_from_slice(symbol.as_bytes());
        bytes.extend_from_slice(&(words.len() as u32).to_le_bytes());
        bytes.extend(words.iter().flat_map(|word| word.to_le_bytes()));
    }
    bytes
}

/// The modules of packed bytes, or `None` if they are malformed.
fn unpack(mut bytes: &[u8]) -> Option<Vec<(String, Vec<u32>)>> {
    fn length(bytes: &mut &[u8]) -> Option<usize> {
        let word: [u8; 4] = bytes.get(..4)?.try_into().ok()?;
        *bytes = bytes.get(4..)?;
        Some(u32::from_le_bytes(word) as usize)
    }
    let mut modules = Vec::new();
    while !bytes.is_empty() {
        let symbol_length = length(&mut bytes)?;
        let symbol = std::str::from_utf8(bytes.get(..symbol_length)?)
            .ok()?
            .to_owned();
        bytes = bytes.get(symbol_length..)?;
        let word_count = length(&mut bytes)?;
        let words = bytes
            .get(..word_count * 4)?
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().expect("four bytes")))
            .collect();
        bytes = bytes.get(word_count * 4..)?;
        modules.push((symbol, words));
    }
    Some(modules)
}

impl Toolchain for VulkanToolchain {
    type Program = DirectModule;

    fn identity(&self) -> &ToolchainIdentity {
        &self.identity
    }

    /// Kept SPIR-V of every requested entry that validates is used as it
    /// is: no compiler runs. The device still creates each pipeline.
    fn keeps(&self, source: &ProgramSource, kept: &[u8]) -> bool {
        self.load(source, kept).is_some()
    }

    fn compile(
        &self,
        source: &ProgramSource,
        cache: Option<&dyn ProgramCache>,
    ) -> Result<DirectModule, NativeCompilationError> {
        let bytes = cache.and_then(|cache| cache.get());
        let pipelines = bytes
            .as_deref()
            .and_then(unpack)
            .and_then(|modules| modules.into_iter().find(|(symbol, _)| symbol == PIPELINES))
            .map_or_else(Vec::new, |(_, words)| pipelines_data(&words));
        let kept = bytes.and_then(|kept| self.load(source, &kept));
        // What is kept is written again whenever any of it was missing: the
        // modules, or the pipelines of modules kept before them.
        let rewrite = kept.is_none() || pipelines.is_empty();
        let modules = match kept {
            Some(modules) => modules,
            None => {
                let environment = self.device.facts().environment();
                source
                    .entries
                    .iter()
                    .map(|entry| formation::compile(&source.text, &entry.symbol, environment))
                    .collect::<Result<Vec<_>, _>>()?
            }
        };
        let entries = modules
            .iter()
            .map(Vec::as_slice)
            .zip(&source.entries)
            .collect::<Vec<_>>();
        let (program, created) = DirectModule::new(&self.device, &entries, &pipelines)?;
        if let Some(cache) = cache.filter(|_| rewrite) {
            let mut kept = source
                .entries
                .iter()
                .map(|entry| entry.symbol.as_str())
                .zip(modules.iter().cloned())
                .collect::<Vec<_>>();
            kept.push((PIPELINES, pipelines_words(&created)));
            cache.put(&pack(&kept));
        }
        Ok(program)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_data_of_any_length_comes_back_as_it_went() {
        for data in [&b""[..], b"a", b"abcd", b"abcdefg"] {
            assert_eq!(pipelines_data(&pipelines_words(data)), data);
        }
    }

    #[test]
    fn packed_modules_keep_their_symbols() {
        let packed = pack(&[("first", vec![1, 2, 3]), ("second", vec![])]);
        assert_eq!(
            unpack(&packed),
            Some(vec![
                ("first".to_owned(), vec![1, 2, 3]),
                ("second".to_owned(), vec![])
            ])
        );
        assert_eq!(unpack(&packed[..packed.len() - 1]), None);
        assert_eq!(unpack(b"xy"), None);
    }
}
