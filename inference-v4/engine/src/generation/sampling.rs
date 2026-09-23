//! Device vocabulary selection. Only the selected token crosses back to the
//! host. The sampler is prepared with the decoder; sampling never compiles.
use super::Sampling;
use crate::{inputs::TokenId, kernels};
use seismic::{Device, Element, Kernel, PreparationOptions, Tensor};

use crate::Error;

pub struct Sampler {
    kernel: Kernel<kernels::sample_rows::Entry>,
    vocabulary: usize,
    mask: Tensor,
    draw: Tensor,
    output: Tensor,
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
    pub fn compile(
        device: &Device,
        vocabulary: usize,
        preparation: PreparationOptions,
    ) -> Result<Self, Error> {
        if vocabulary == 0 || vocabulary > i32::MAX as usize {
            return Err("sampling vocabulary is outside the index domain".into());
        }
        let words = u64::try_from(vocabulary.div_ceil(32))
            .map_err(|_| "sampling mask extent exceeds the Seismic shape domain")?;
        Ok(Self {
            kernel: kernels::sample_rows::for_device(device, preparation)?,
            vocabulary,
            mask: Tensor::zeros(device, Element::u32(), &[1, words])?,
            draw: Tensor::zeros(device, Element::u32(), &[1, 6])?,
            output: Tensor::zeros(device, Element::i32(), &[1, 2])?,
        })
    }

    /// Synchronous completion includes mask transfer, selection, and token readback.
    /// Call only after the producer of `logits` has completed.
    pub fn sample(
        &mut self,
        logits: &Tensor,
        mask: Option<&[u32]>,
        sampling: Sampling,
        seed: u64,
        position: usize,
    ) -> Result<Selection, Error> {
        let words = self.vocabulary.div_ceil(32);
        if mask.is_some_and(|mask| mask.len() != words) {
            return Err("sampling mask has the wrong vocabulary extent".into());
        }
        let bytes: Vec<_> = (0..words)
            .flat_map(|i| mask.map_or(u32::MAX, |mask| mask[i]).to_le_bytes())
            .collect();
        self.mask.write_from_host(&bytes)?;
        let position = u64::try_from(position)
            .map_err(|_| "sampling position exceeds the Seismic scalar domain")?;
        let draw = [
            u32::from(sampling == Sampling::Categorical),
            seed as u32,
            (seed >> 32) as u32,
            position as u32,
            (position >> 32) as u32,
            0,
        ];
        self.draw.write_from_host(
            &draw
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )?;
        self.kernel.call(kernels::sample_rows::Args {
            logits,
            mask: &self.mask,
            draws: &self.draw,
            result: &mut self.output,
        })?;
        let bytes = self.output.read_to_host()?;
        let selected = i32::from_le_bytes(bytes[..4].try_into().unwrap());
        match i32::from_le_bytes(bytes[4..].try_into().unwrap()) {
            0 => {}
            1 => return Ok(Selection::Empty),
            2 => return Ok(Selection::Nonfinite),
            status => panic!("sample_rows returned impossible status {status}"),
        }
        if selected < 0 || selected as usize >= self.vocabulary {
            panic!("sample_rows returned an out-of-domain successful token");
        }
        Ok(Selection::Token(TokenId(selected as u32)))
    }
}
