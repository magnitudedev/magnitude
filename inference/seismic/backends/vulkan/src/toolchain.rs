//! Vulkan's program toolchain.
//!
//! Compiling a program compiles its source once per entry (the entry is
//! selected by `SEISMIC_KERNEL`) to sealed, validated SPIR-V, then binds
//! `fma` for the device and creates each entry's pipeline with its constants
//! through the device's pipeline cache. It keeps each entry's SPIR-V: a kept
//! module is used when it validates, and a program whose modules are not all
//! kept is compiled again.

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

    fn compile(
        &self,
        source: &ProgramSource,
        cache: Option<&dyn ProgramCache>,
    ) -> Result<DirectModule, NativeCompilationError> {
        let kept = cache
            .and_then(|cache| cache.get())
            .and_then(|kept| self.load(source, &kept));
        let modules = match kept {
            Some(modules) => modules,
            None => {
                let environment = self.device.facts().environment();
                let modules = source
                    .entries
                    .iter()
                    .map(|entry| formation::compile(&source.text, &entry.symbol, environment))
                    .collect::<Result<Vec<_>, _>>()?;
                if let Some(cache) = cache {
                    cache.put(&pack(
                        &source
                            .entries
                            .iter()
                            .map(|entry| entry.symbol.as_str())
                            .zip(modules.iter().cloned())
                            .collect::<Vec<_>>(),
                    ));
                }
                modules
            }
        };
        let entries = modules
            .iter()
            .map(Vec::as_slice)
            .zip(&source.entries)
            .collect::<Vec<_>>();
        DirectModule::new(&self.device, &entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
