//! Explicit conditioned-input preparation (package E1 input phase). The cast
//! overlays for one conditioned forward are compiled here, in full, before the
//! caller builds any numerical inference submission. All compilation goes
//! through the preparation session; this module never names the compiler.
use crate::models::qwen35::inputs::Assembled;
use crate::preparation::{
    CompositionSpec, EnvelopeShape, PreparedComposition, PreparationSession, Program, Settings,
    WorkloadEnvelope,
};
use seismic_lang::types::{DType, Elem};
use seismic_runtime::{Buffer, Device};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    rc::Rc,
};

use crate::Error;

/// One prepared conditioning overlay: a sealed cast composition plus the
/// destination slice it writes into the embedded hidden rows.
pub(super) struct PreparedOverlay {
    pub(super) composition: PreparedComposition,
    pub(super) count: usize,
    pub(super) source: Buffer,
    pub(super) offset: usize,
    pub(super) length: usize,
}

pub(super) struct Conditioning {
    device: Rc<Device>,
    program: Program,
    settings: Settings,
    hidden: u64,
}

impl Conditioning {
    pub(super) fn new(device: Rc<Device>, program: Program, settings: Settings, hidden: u64) -> Self {
        Self {
            device,
            program,
            settings,
            hidden,
        }
    }

    /// Compile the cast overlays for one assembled conditioned input. The
    /// input-preparation phase: it completes before any numerical inference
    /// submission is built.
    pub(super) fn prepare(&self, assembled: &Assembled) -> Result<Vec<PreparedOverlay>, Error> {
        let row_bytes = usize::try_from(self.hidden)
            .map_err(|_| "hidden width overflow")?
            .checked_mul(4)
            .ok_or("hidden row bytes overflow")?;
        let mut session = PreparationSession::new(&self.device, &self.program, self.settings.clone());
        let mut overlays = Vec::with_capacity(assembled.features.len());
        for feature in &assembled.features {
            let offset = feature
                .destination
                .checked_mul(row_bytes)
                .ok_or("feature destination overflow")?;
            let length = feature
                .count
                .checked_mul(row_bytes)
                .ok_or("feature destination overflow")?;
            let envelope = WorkloadEnvelope::new(
                BTreeMap::from([
                    ("M".into(), EnvelopeShape::Exact(feature.count as u64)),
                    ("K".into(), EnvelopeShape::Exact(self.hidden)),
                ]),
                BTreeMap::from([
                    ("T".into(), Elem::Dtype(DType::F32)),
                    ("U".into(), Elem::Dtype(DType::F32)),
                ]),
                Vec::new(),
            )?;
            let composition = session.prepare(CompositionSpec {
                entry: "cast_rows".into(),
                envelope,
                weights: HashMap::new(),
                external: HashSet::from(["input".into(), "out".into()]),
                intermediates: HashSet::new(),
                scalars: HashMap::new(),
            })?;
            overlays.push(PreparedOverlay {
                composition,
                count: feature.count,
                source: feature.source.clone(),
                offset,
                length,
            });
        }
        Ok(overlays)
    }
}
