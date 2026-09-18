//! Device vocabulary selection. Only the selected token crosses back to the host.
use super::Sampling;
use crate::{
    execution::{Composition, CompositionSpec},
    inputs::TokenId,
};
use seismic_runtime::{
    plan::{PlanCompiler, Settings},
    Buffer, Device, Error,
};
use std::collections::{HashMap, HashSet};

pub struct Sampler {
    composition: Composition,
    vocabulary: usize,
    mask: Buffer,
    draw: Buffer,
    output: Buffer,
}
/// A completed selection can reject one request without failing device execution
/// or the accepted progress of its peers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selection {
    Token(TokenId),
    Empty,
    Nonfinite,
}
impl Sampler {
    pub(crate) fn kernel_count(&self) -> usize {
        self.composition.kernel_count()
    }
    pub fn compile(device: &Device, vocabulary: usize, settings: Settings) -> Result<Self, Error> {
        if vocabulary == 0 || vocabulary > i32::MAX as usize {
            return Err("sampling vocabulary is outside the index domain".into());
        }
        let program = seismic_std::program()?;
        let mut compiler = PlanCompiler::new(device, &program, settings);
        let composition = Composition::compile(
            &mut compiler,
            CompositionSpec {
                entry: "sample_rows".into(),
                shapes: HashMap::from([("M".into(), 1), ("V".into(), vocabulary as i64)]),
                elements: HashMap::new(),
                weights: HashMap::new(),
                external: ["logits", "mask", "draws", "out"]
                    .into_iter()
                    .map(String::from)
                    .collect(),
                intermediates: HashSet::new(),
                scalars: HashMap::new(),
            },
        )?;
        Ok(Self {
            composition,
            vocabulary,
            mask: device.buffer(vocabulary.div_ceil(32) * 4)?,
            draw: device.buffer(24)?,
            output: device.buffer(8)?,
        })
    }

    /// Synchronous completion includes mask transfer, selection, and token readback.
    /// Call only after the producer of `logits` has completed.
    pub fn sample(
        &mut self,
        logits: &Buffer,
        mask: Option<&[u32]>,
        sampling: Sampling,
        seed: u64,
        position: usize,
    ) -> Result<Selection, Error> {
        if logits.len() != self.vocabulary * 4 {
            return Err("sampling logits have the wrong vocabulary extent".into());
        }
        let words = self.vocabulary.div_ceil(32);
        if mask.is_some_and(|mask| mask.len() != words) {
            return Err("sampling mask has the wrong vocabulary extent".into());
        }
        let bytes: Vec<_> = (0..words)
            .flat_map(|i| mask.map_or(u32::MAX, |mask| mask[i]).to_le_bytes())
            .collect();
        self.mask.write(&bytes)?;
        let position = position as u64;
        let draw = [
            u32::from(sampling == Sampling::Categorical),
            seed as u32,
            (seed >> 32) as u32,
            position as u32,
            (position >> 32) as u32,
            0,
        ];
        self.draw.write(
            &draw
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )?;
        self.composition.execute(
            &HashMap::from([
                ("logits".into(), logits.clone()),
                ("mask".into(), self.mask.clone()),
                ("draws".into(), self.draw.clone()),
                ("out".into(), self.output.clone()),
            ]),
            &HashMap::new(),
        )?;
        let mut bytes = [0; 8];
        self.output.read(&mut bytes)?;
        let selected = i32::from_le_bytes(bytes[..4].try_into().unwrap());
        match i32::from_le_bytes(bytes[4..].try_into().unwrap()) {
            0 => {}
            1 => return Ok(Selection::Empty),
            2 => return Ok(Selection::Nonfinite),
            _ => return Err("unknown sampling status".into()),
        }
        if selected < 0 || selected as usize >= self.vocabulary {
            return Err("invalid or empty sampling distribution".into());
        }
        Ok(Selection::Token(TokenId(selected as u32)))
    }
}
