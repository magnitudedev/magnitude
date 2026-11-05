//! CUDA's program toolchain.
//!
//! Compiling a program runs NVRTC once for the device's architecture with
//! every entry symbol as a name expression, so plain `extern "C"` kernels
//! and template instances are requested alike. It keeps the CUBIN with each
//! requested expression's lowered linker name, which NVRTC alone supplies:
//! a kept image serves any entries it holds, and one the driver refuses, or
//! that lacks an entry, is compiled again.

use crate::direct::{formation_error, DirectModule};
use crate::executor::Device;
use crate::nvrtc;
use seismic_native_target::{
    Artifact, Load, NativeCompilationError, ProgramEntry, ProgramSource, Toolchain,
    ToolchainIdentity,
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
            },
        }
    }

}

/// Everything besides the source that determines a CUBIN: the release,
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

impl Artifact for Cubin {
    const ENCODING: u32 = 1;

    fn encode(&self) -> Vec<u8> {
        let expressions = self.names.iter().map(|(expression, _)| expression.as_str()).collect::<Vec<_>>();
        let lowered = self.names.iter().map(|(_, lowered)| lowered.clone()).collect::<Vec<_>>();
        pack(&expressions, &lowered, &self.image)
    }

    fn decode(bytes: &[u8]) -> Option<Self> {
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
    /// Loading a CUBIN compiles nothing.
    const LOAD: Load = Load::Complete;

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

    fn load(
        &self,
        artifact: &Cubin,
        entries: &[ProgramEntry],
    ) -> Result<DirectModule, NativeCompilationError> {
        let kernels = entries
            .iter()
            .map(|entry| {
                artifact
                    .names
                    .iter()
                    .find(|(expression, _)| *expression == entry.symbol)
                    .map(|(_, lowered)| lowered.as_str())
                    .ok_or_else(|| {
                        NativeCompilationError::MalformedToolchainOutput(format!(
                            "the CUBIN holds no `{}`",
                            entry.symbol
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        DirectModule::load(&self.device, &artifact.image, &kernels)
    }
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
