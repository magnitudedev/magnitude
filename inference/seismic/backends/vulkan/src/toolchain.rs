//! Vulkan's program toolchain.
//!
//! A program's artifact is each entry's SPIR-V, compiled from its source
//! once per entry (the entry is selected by `SEISMIC_KERNEL`) and sealed and
//! validated, with the pipeline cache data of creating each entry's
//! pipeline with its constants on this device and driver. Loading creates
//! the pipelines from that data and compiles nothing. A compile hands back
//! the pipelines it created, and first creates them again from its data, so
//! an artifact whose data would not serve a load fails its compile rather
//! than being stored and failing every later load. Where the device
//! reports pipeline statistics, every pipeline, a load's as a compile's,
//! is created capturing them, and the program a compile or load hands back
//! reads them from the driver, so the artifact carries none.

use crate::device::Device;
use crate::formation::{self, DirectModule};
use seismic_native_target::{
    Artifact, Load, NativeCompilationError, ProgramEntry, ProgramSource, ProgramStatistic,
    Toolchain, ToolchainIdentity,
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
}

/// A Vulkan program's artifact: each entry's sealed SPIR-V, and the
/// pipeline cache data creating every entry's pipeline left on this device
/// and driver, which pipelines are created from without compiling.
pub struct SpirvPipelines {
    modules: Vec<(String, Vec<u32>)>,
    pipelines: Vec<u8>,
}

impl Artifact for SpirvPipelines {
    const ENCODING: u32 = 1;

    /// The pipeline cache data, length-prefixed, then the modules
    /// ([`pack`]).
    fn encode(&self) -> Vec<u8> {
        let length = u32::try_from(self.pipelines.len()).expect("pipeline cache data fits u32");
        let mut bytes = length.to_le_bytes().to_vec();
        bytes.extend_from_slice(&self.pipelines);
        bytes.extend(pack(&self.modules));
        bytes
    }

    fn decode(bytes: &[u8]) -> Option<Self> {
        let length = u32::from_le_bytes(bytes.get(..4)?.try_into().ok()?) as usize;
        let end = 4usize.checked_add(length)?;
        Some(Self {
            pipelines: bytes.get(4..end)?.to_vec(),
            modules: unpack(bytes.get(end..)?)?,
        })
    }
}

/// Each module's symbol and words, length-prefixed, little-endian.
fn pack<S: AsRef<str>>(modules: &[(S, Vec<u32>)]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for (symbol, words) in modules {
        let symbol = symbol.as_ref();
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
    type Artifact = SpirvPipelines;
    type Program = DirectModule;
    /// Pipelines are created from the artifact's data and fail where it
    /// would take a compile.
    const LOAD: Load = Load::Complete;

    fn identity(&self) -> &ToolchainIdentity {
        &self.identity
    }

    fn compile(
        &self,
        source: &ProgramSource,
    ) -> Result<(SpirvPipelines, DirectModule), NativeCompilationError> {
        let environment = self.device.facts().environment();
        let modules = source
            .entries
            .iter()
            .map(|entry| formation::compile(&source.text, &entry.symbol, environment))
            .collect::<Result<Vec<_>, _>>()?;
        let entries = modules
            .iter()
            .map(Vec::as_slice)
            .zip(&source.entries)
            .collect::<Vec<_>>();
        let (program, pipelines) = DirectModule::compile(&self.device, &entries)?;
        DirectModule::serves(&self.device, &entries, &pipelines)?;
        let artifact = SpirvPipelines {
            modules: source
                .entries
                .iter()
                .map(|entry| entry.symbol.clone())
                .zip(modules)
                .collect(),
            pipelines,
        };
        Ok((artifact, program))
    }

    fn load(
        &self,
        artifact: &SpirvPipelines,
        entries: &[ProgramEntry],
    ) -> Result<DirectModule, NativeCompilationError> {
        let held = entries
            .iter()
            .map(|entry| {
                artifact
                    .modules
                    .iter()
                    .find(|(symbol, _)| *symbol == entry.symbol)
                    .map(|(_, words)| (words.as_slice(), entry))
                    .ok_or_else(|| {
                        NativeCompilationError::MalformedToolchainOutput(format!(
                            "the Vulkan artifact holds no `{}`",
                            entry.symbol
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        DirectModule::load(&self.device, &held, &artifact.pipelines)
    }

    fn statistics(program: &DirectModule, entry: usize) -> &[ProgramStatistic] {
        program.statistics(entry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A program compiled once loads from its artifact in another
    /// toolchain on the same device without compiling: its pipelines are
    /// created with `FAIL_ON_PIPELINE_COMPILE_REQUIRED`, which fails where
    /// the artifact's data would not serve them.
    #[test]
    fn a_program_loads_from_its_artifact_without_compiling() {
        for description in crate::test_devices() {
            let open = || {
                Device::open(description.facts.uuid, crate::test_accounts(&description.facts))
                    .expect("a device meeting the floor opens")
            };
            let source = ProgramSource {
                text: "#version 460\nlayout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;\n\
                       shared float x[64];\n\
                       void main() { x[gl_LocalInvocationID.x] = 1.0; }\n"
                    .into(),
                entries: vec![ProgramEntry {
                    symbol: "main".into(),
                    group_size: Some([64, 1, 1]),
                    constants: Vec::new(),
                }],
            };
            let bytes = VulkanToolchain::new(open())
                .compile(&source)
                .expect("the program compiles")
                .0
                .encode();
            let artifact = SpirvPipelines::decode(&bytes).expect("the artifact decodes");
            VulkanToolchain::new(open())
                .load(&artifact, &source.entries)
                .expect("the artifact serves the program without compiling");
        }
    }

    #[test]
    fn an_artifact_of_any_pipeline_data_length_comes_back_as_it_went() {
        for data in [&b""[..], b"a", b"abcd", b"abcdefg"] {
            let artifact = SpirvPipelines {
                modules: vec![("main".into(), vec![7, 8])],
                pipelines: data.to_vec(),
            };
            let decoded = SpirvPipelines::decode(&artifact.encode()).unwrap();
            assert_eq!(decoded.pipelines, data);
            assert_eq!(decoded.modules, artifact.modules);
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
