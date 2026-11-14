//! CUDA's program toolchains, one a source language: CUDA C++
//! ([`CudaToolchain`]) and PTX ([`CudaPtxToolchain`]). Both keep a CUBIN for
//! the device's architecture, which a load hands the driver: no compiler
//! runs at a load.
//!
//! Compiling a CUDA C++ program runs NVRTC once for the device's architecture with
//! every entry symbol as a name expression, so plain `extern "C"` kernels
//! and template instances are requested alike. It keeps the CUBIN with each
//! requested expression's lowered linker name, which NVRTC alone supplies:
//! a kept image serves any entries it holds, and one the driver refuses, or
//! that lacks an entry, is declined ([`Declined`]): a miss, compiled again.

use crate::direct::{formation_error, jit_report, DirectModule};
use crate::driver::{compile_image, JitError};
use crate::executor::Device;
use crate::nvrtc;
use seismic_native_target::{
    Artifact, Declined, NativeCompilationError, ProgramEntry, ProgramSource, ProgramStatistic,
    Toolchain, ToolchainIdentity,
};

pub struct CudaToolchain {
    device: Device,
    architecture: u32,
    /// The NVRTC formation for the device's architecture, or why there is
    /// none: every compile then fails with it.
    formation: Result<nvrtc::Formation, NativeCompilationError>,
    identity: ToolchainIdentity,
}

impl CudaToolchain {
    pub fn new(device: Device, architecture: u32) -> Self {
        let formation = nvrtc::formation(architecture).map_err(formation_error);
        let material = match &formation {
            Ok(formation) => material(formation),
            Err(error) => format!("cuda;sm_{architecture};unavailable: {error:?}"),
        };
        Self {
            device,
            architecture,
            formation,
            identity: ToolchainIdentity {
                namespace: "cuda",
                material,
                stored: format!("build {FORMATION_BUILD}"),
            },
        }
    }

}

/// This build's formation code ([`build.rs`](../build.rs)): part of every
/// stored artifact's key, never of tuning's.
pub const FORMATION_BUILD: &str = env!("SEISMIC_FORMATION_BUILD");

/// Everything besides the source and this build that determines a CUBIN: the release,
/// architecture and options of `formation`. The driver only loads it.
pub fn material(formation: &nvrtc::Formation) -> String {
    format!("cuda;{formation}")
}

/// A CUDA program's artifact: its CUBIN, the architecture's final code,
/// with each entry's name expression and lowered name.
pub struct Cubin {
    names: Vec<(String, String)>,
    image: Vec<u8>,
}

impl Cubin {
    /// Its bytes, as [`Artifact::decode`] reads them ([`pack`]).
    fn encode(&self) -> Vec<u8> {
        let expressions = self.names.iter().map(|(expression, _)| expression.as_str()).collect::<Vec<_>>();
        let lowered = self.names.iter().map(|(_, lowered)| lowered.clone()).collect::<Vec<_>>();
        pack(&expressions, &lowered, &self.image)
    }
}

impl Artifact for Cubin {
    const ENCODING: u32 = 1;

    fn decode(bytes: &[u8], _: Option<&std::path::Path>) -> Option<Self> {
        let (names, image) = unpack(bytes)?;
        Some(Self {
            names,
            image: image.to_vec(),
        })
    }
}

impl Toolchain for CudaToolchain {
    type Artifact = Cubin;
    type Program = DirectModule;
    /// The artifact itself: the compile made it.
    type Pending = Cubin;

    fn identity(&self) -> &ToolchainIdentity {
        &self.identity
    }

    fn compile(
        &self,
        source: &ProgramSource,
    ) -> Result<(Cubin, DirectModule), NativeCompilationError> {
        self.formation.as_ref().map_err(Clone::clone)?;
        let cubin = form_cubin(source, self.architecture)?;
        let kernels = cubin.lowered_names.iter().map(String::as_str).collect::<Vec<_>>();
        let program = DirectModule::load(&self.device, &cubin.image, &kernels)?;
        let artifact = Cubin {
            names: source
                .entries
                .iter()
                .map(|entry| entry.symbol.clone())
                .zip(cubin.lowered_names)
                .collect(),
            image: cubin.image,
        };
        Ok((artifact, program))
    }

    fn artifact(&self, artifact: Cubin) -> Result<Vec<u8>, NativeCompilationError> {
        Ok(artifact.encode())
    }

    /// A CUBIN without a requested entry, or one the driver refuses, is
    /// declined: a miss, compiled again.
    fn load(&self, artifact: &Cubin, entries: &[ProgramEntry]) -> Result<DirectModule, Declined> {
        let kernels = entries
            .iter()
            .map(|entry| {
                artifact
                    .names
                    .iter()
                    .find(|(expression, _)| *expression == entry.symbol)
                    .map(|(_, lowered)| lowered.as_str())
                    .ok_or_else(|| Declined(format!("the CUBIN holds no `{}`", entry.symbol)))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(DirectModule::load(&self.device, &artifact.image, &kernels)?)
    }

    fn statistics(program: &DirectModule, entry: usize) -> &[ProgramStatistic] {
        program.statistics(entry)
    }
}

/// The toolchain of PTX programs (Seismic-IR kernels, the device's
/// graph-cost and analytical-service probe kernels): the driver's JIT
/// links the text to a CUBIN for the device's architecture once, and every
/// later load hands the driver that CUBIN, so no PTX reaches the driver at
/// a load. A PTX entry's name is its symbol: the CUBIN is kept with each
/// name paired with itself.
pub struct CudaPtxToolchain {
    device: Device,
    identity: ToolchainIdentity,
}

impl CudaPtxToolchain {
    pub fn new(device: Device, architecture: u32) -> Self {
        let version = device
            .context()
            .driver
            .version()
            .expect("the driver of an opened device reports its version");
        Self {
            device,
            identity: ToolchainIdentity {
                namespace: "cuda-ptx",
                material: ptx_material(version, architecture),
                stored: format!("build {FORMATION_BUILD}"),
            },
        }
    }
}

/// Everything besides the source and this build (whose formation code
/// holds the JIT's options) that determines a PTX program's CUBIN: the
/// driver, whose JIT compiles it, and the architecture it targets.
fn ptx_material(driver_version: i32, architecture: u32) -> String {
    format!("cuda-ptx;driver {driver_version};sm_{architecture}")
}

impl Toolchain for CudaPtxToolchain {
    type Artifact = Cubin;
    type Program = DirectModule;
    /// The artifact itself: the compile made it.
    type Pending = Cubin;

    fn identity(&self) -> &ToolchainIdentity {
        &self.identity
    }

    fn compile(&self, source: &ProgramSource) -> Result<(Cubin, DirectModule), NativeCompilationError> {
        let image = compile_image(self.device.context(), &source.text).map_err(|error| ptx_failure(error, &source.text))?;
        let kernels = source.entries.iter().map(|entry| entry.symbol.as_str()).collect::<Vec<_>>();
        let program = DirectModule::load(&self.device, &image, &kernels)?;
        let artifact = Cubin {
            names: source.entries.iter().map(|entry| (entry.symbol.clone(), entry.symbol.clone())).collect(),
            image,
        };
        Ok((artifact, program))
    }

    fn artifact(&self, artifact: Cubin) -> Result<Vec<u8>, NativeCompilationError> {
        Ok(artifact.encode())
    }

    /// A CUBIN without a requested entry, or one the driver refuses, is
    /// declined: a miss, compiled again.
    fn load(&self, artifact: &Cubin, entries: &[ProgramEntry]) -> Result<DirectModule, Declined> {
        let kernels = entries
            .iter()
            .map(|entry| {
                artifact
                    .names
                    .iter()
                    .any(|(name, _)| *name == entry.symbol)
                    .then_some(entry.symbol.as_str())
                    .ok_or_else(|| Declined(format!("the CUBIN holds no `{}`", entry.symbol)))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(DirectModule::load(&self.device, &artifact.image, &kernels)?)
    }

    fn statistics(program: &DirectModule, entry: usize) -> &[ProgramStatistic] {
        program.statistics(entry)
    }
}

/// The JIT's failure on `ptx`, with the text's first lines, numbered as its
/// log numbers them.
fn ptx_failure(error: JitError, ptx: &str) -> NativeCompilationError {
    let numbered = ptx
        .lines()
        .enumerate()
        .take(160)
        .map(|(line, text)| format!("{:>4}: {text}", line + 1))
        .collect::<Vec<_>>()
        .join("\n");
    NativeCompilationError::ToolchainFailure(format!("{}\nGenerated PTX (first 160 lines):\n{numbered}", jit_report(error)))
}

/// `source`'s CUBIN for `sm_<architecture>`, each entry requested by its
/// name expression: everything a CUDA program is formed from before the
/// driver loads it.
pub fn form_cubin(
    source: &ProgramSource,
    architecture: u32,
) -> Result<nvrtc::Cubin, NativeCompilationError> {
    let expressions = source
        .entries
        .iter()
        .map(|entry| entry.symbol.as_str())
        .collect::<Vec<_>>();
    let name = format!("{}.cu", expressions.first().copied().unwrap_or("program"));
    nvrtc::compile_cubin_named(&source.text, &name, architecture, &expressions)
        .map_err(formation_error)
}

const MAGIC: &[u8; 8] = b"SCUNAM02";

/// The magic, the entry count, each entry's expression and lowered name
/// (length-prefixed), then the CUBIN.
fn pack(expressions: &[&str], lowered: &[String], cubin: &[u8]) -> Vec<u8> {
    let mut image = MAGIC.to_vec();
    let count = u32::try_from(expressions.len()).expect("entry count fits u32");
    image.extend_from_slice(&count.to_le_bytes());
    for (expression, lowered) in expressions.iter().zip(lowered) {
        for text in [*expression, lowered.as_str()] {
            image.extend_from_slice(&(text.len() as u32).to_le_bytes());
            image.extend_from_slice(text.as_bytes());
        }
    }
    image.extend_from_slice(cubin);
    image
}

/// The (expression, lowered name) pairs and CUBIN of a packed image, or
/// `None` if it is malformed.
fn unpack(image: &[u8]) -> Option<(Vec<(String, String)>, &[u8])> {
    fn length(remaining: &mut &[u8]) -> Option<usize> {
        let bytes: [u8; 4] = remaining.get(..4)?.try_into().ok()?;
        *remaining = remaining.get(4..)?;
        Some(u32::from_le_bytes(bytes) as usize)
    }
    fn text(remaining: &mut &[u8]) -> Option<String> {
        let length = length(remaining)?;
        let text = std::str::from_utf8(remaining.get(..length)?)
            .ok()?
            .to_owned();
        *remaining = remaining.get(length..)?;
        Some(text)
    }
    let mut remaining = image.strip_prefix(MAGIC)?;
    let count = length(&mut remaining)?;
    let names = (0..count)
        .map(|_| Some((text(&mut remaining)?, text(&mut remaining)?)))
        .collect::<Option<Vec<_>>>()?;
    (!remaining.is_empty()).then_some((names, remaining))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_packed_image_keeps_each_expressions_lowered_name() {
        let lowered = vec![
            "_Z5probeILi2EEvPf".to_owned(),
            "_Z5probeILi4EEvPf".to_owned(),
        ];
        let image = pack(&["probe<2>", "probe<4>"], &lowered, b"cubin");
        let (names, cubin) = unpack(&image).unwrap();
        assert_eq!(cubin, b"cubin");
        assert_eq!(
            names,
            vec![
                ("probe<2>".to_owned(), lowered[0].clone()),
                ("probe<4>".to_owned(), lowered[1].clone()),
            ]
        );
        assert_eq!(unpack(&image[..image.len() - 5]), None);
        assert_eq!(unpack(b"not an image"), None);
    }
}
