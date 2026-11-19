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
    Artifact, Declined, NativeCompilationError, ProgramEntry, ProgramSource, ProgramStatistic,
    Toolchain, ToolchainIdentity,
};

/// This build's formation code ([`build.rs`](../build.rs)): its seal and
/// validation passes, its glslang and SPIR-V Tools, and how it creates
/// pipelines. Part of every stored artifact's key, never of tuning's.
pub const FORMATION_BUILD: &str = env!("SEISMIC_FORMATION_BUILD");

pub struct VulkanToolchain {
    device: Device,
    identity: ToolchainIdentity,
}

impl VulkanToolchain {
    pub fn new(device: Device) -> Self {
        let identity = ToolchainIdentity {
            namespace: "vulkan",
            material: formation::formation(&device),
            stored: format!("build {FORMATION_BUILD}"),
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

impl SpirvPipelines {
    /// Its bytes, as [`Artifact::decode`] reads them: the pipeline cache
    /// data, length-prefixed, then the modules ([`pack`]).
    fn encode(&self) -> Vec<u8> {
        let length = u32::try_from(self.pipelines.len()).expect("pipeline cache data fits u32");
        let mut bytes = length.to_le_bytes().to_vec();
        bytes.extend_from_slice(&self.pipelines);
        bytes.extend(pack(&self.modules));
        bytes
    }
}

impl Artifact for SpirvPipelines {
    const ENCODING: u32 = 1;

    fn decode(bytes: &[u8], _: Option<seismic_native_target::HeldFile>) -> Option<Self> {
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
    /// The artifact itself: the compile made it.
    type Pending = SpirvPipelines;

    fn identity(&self) -> &ToolchainIdentity {
        &self.identity
    }

    fn compile(
        &self,
        source: &ProgramSource,
    ) -> Result<(SpirvPipelines, DirectModule), NativeCompilationError> {
        let environment = self.device.formation().environment();
        let modules = source
            .entries
            .iter()
            .map(|entry| formation::form(&source.text, entry, environment))
            .collect::<Result<Vec<_>, _>>()?;
        let entries = modules
            .iter()
            .map(Vec::as_slice)
            .zip(&source.entries)
            .collect::<Vec<_>>();
        let (program, pipelines) = DirectModule::compile(&self.device, &entries)?;
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

    fn artifact(&self, artifact: SpirvPipelines) -> Result<Vec<u8>, NativeCompilationError> {
        Ok(artifact.encode())
    }

    /// An artifact without a requested entry, or whose pipeline cache data
    /// the driver would have to compile past, is declined: a miss,
    /// compiled again.
    fn load(&self, artifact: &SpirvPipelines, entries: &[ProgramEntry]) -> Result<DirectModule, Declined> {
        let held = entries
            .iter()
            .map(|entry| {
                artifact
                    .modules
                    .iter()
                    .find(|(symbol, _)| *symbol == entry.symbol)
                    .map(|(_, words)| (words.as_slice(), entry))
                    .ok_or_else(|| Declined(format!("the Vulkan artifact holds no `{}`", entry.symbol)))
            })
            .collect::<Result<Vec<_>, _>>()?;
        DirectModule::load(&self.device, &held, &artifact.pipelines)?
            .ok_or_else(|| Declined("the driver does not create a pipeline from the stored cache data without compiling".into()))
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
    /// created with `FAIL_ON_PIPELINE_COMPILE_REQUIRED`, so the load either
    /// holds every pipeline or is declined for the driver's own reason
    /// (it would compile), never for the artifact's.
    #[test]
    fn a_program_loads_from_its_artifact_or_misses() {
        for description in crate::test_devices() {
            let open = || {
                Device::open(description.facts.uuid, crate::unbounded_stores(&description.facts), &Default::default())
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
            let artifact = SpirvPipelines::decode(&bytes, None).expect("the artifact decodes");
            if let Err(declined) = VulkanToolchain::new(open()).load(&artifact, &source.entries) {
                assert!(declined.0.contains("without compiling"), "a load is served or the driver declines its data: {declined}");
            }
        }
    }

    #[test]
    fn an_artifact_of_any_pipeline_data_length_comes_back_as_it_went() {
        for data in [&b""[..], b"a", b"abcd", b"abcdefg"] {
            let artifact = SpirvPipelines {
                modules: vec![("main".into(), vec![7, 8])],
                pipelines: data.to_vec(),
            };
            let decoded = SpirvPipelines::decode(&artifact.encode(), None).unwrap();
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
