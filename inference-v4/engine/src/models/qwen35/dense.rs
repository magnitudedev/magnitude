//! Prepared dense feedforward suffix of a Qwen block. `DenseSuffix::compile`
//! is model preparation: it compiles and seals the whole (exact) workload
//! envelope before returning; execution invokes the sealed plan only.
use crate::{
    execution,
    preparation::{
        CompositionSpec, EnvelopeShape, PreparationSession, Program, Settings, WorkloadEnvelope,
    },
    weights::residency::ResidentWeight,
};
use seismic_lang::types::{DType, Elem};
use seismic_runtime::{Buffer, Device};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone)]
pub struct DenseWeights {
    pub norm: ResidentWeight,
    pub gate: ResidentWeight,
    pub up: ResidentWeight,
    pub down: ResidentWeight,
}
#[derive(Clone, Copy)]
pub struct DenseInvocation {
    pub rows: usize,
    pub activation: DType,
    pub epsilon: f32,
}
/// Owns the sealed prepared composition and weights. The compiler/runtime own
/// all result destinations.
pub struct DenseSuffix {
    composition: crate::preparation::PreparedComposition,
    weights: DenseWeights,
    epsilon: f32,
    rows: usize,
}
impl DenseSuffix {
    pub fn compile(
        device: &Device,
        program: &Program,
        invocation: DenseInvocation,
        weights: DenseWeights,
        settings: Settings,
    ) -> Result<Self, String> {
        let DenseInvocation {
            rows,
            activation,
            epsilon,
        } = invocation;
        if rows == 0
            || !matches!(activation, DType::BF16 | DType::F16)
            || !epsilon.is_finite()
            || epsilon <= 0.0
        {
            return Err("dense suffix requires nonempty rows, compact floating activations and positive finite epsilon".into());
        }
        let [hidden] = weights.norm.descriptor().shape.as_slice() else {
            return Err("dense suffix norm must be a vector".into());
        };
        let [intermediate, gate_hidden] = weights.gate.descriptor().shape.as_slice() else {
            return Err("dense suffix gate must be a matrix".into());
        };
        if *hidden == 0
            || *intermediate == 0
            || gate_hidden != hidden
            || weights.up.descriptor().shape != [*intermediate, *hidden]
            || weights.down.descriptor().shape != [*hidden, *intermediate]
        {
            return Err(
                "dense suffix weights disagree on hidden and intermediate dimensions".into(),
            );
        }
        let mut session = PreparationSession::new(device, program, settings);
        let envelope = WorkloadEnvelope::new(
            BTreeMap::from([
                ("M".into(), EnvelopeShape::Exact(rows as u64)),
                ("H".into(), EnvelopeShape::Exact(*hidden)),
                ("F".into(), EnvelopeShape::Exact(*intermediate)),
            ]),
            BTreeMap::from([
                ("A".into(), Elem::Dtype(activation)),
                ("NW".into(), weights.norm.element().clone()),
                ("GW".into(), weights.gate.element().clone()),
                ("UW".into(), weights.up.element().clone()),
                ("DW".into(), weights.down.element().clone()),
            ]),
            Vec::new(),
        )
        .map_err(|e| e.to_string())?;
        let composition = session.prepare(CompositionSpec {
            entry: "qwen_dense_suffix".into(),
            envelope,
            weights: HashMap::from([
                ("norm".into(), weights.norm.clone()),
                ("gate_weight".into(), weights.gate.clone()),
                ("up_weight".into(), weights.up.clone()),
                ("down_weight".into(), weights.down.clone()),
            ]),
            external: HashSet::from(["residual".into()]),
            intermediates: HashSet::new(),
            scalars: HashMap::new(),
        })
        .map_err(|e| e.to_string())?;
        Ok(Self {
            composition,
            weights,
            epsilon,
            rows,
        })
    }
    pub fn kernel_count(&self) -> usize {
        self.composition.kernel_count()
    }
    pub fn execute(&self, residual: &Buffer) -> Result<Buffer, String> {
        let results = execution::execute(
            &self.composition,
            &BTreeMap::from([("M".into(), self.rows as u64)]),
            &HashMap::from([("residual".into(), residual.clone())]),
            &HashMap::from([("eps".into(), f64::from(self.epsilon))]),
        )
        .map_err(|e| e.to_string())?;
        results
            .planes
            .into_iter()
            .find(|result| result.path == [6] && result.plane.is_empty())
            .map(|result| result.buffer)
            .ok_or_else(|| "dense suffix returned no F32 residual result".into())
    }
    pub fn weights(&self) -> &DenseWeights {
        &self.weights
    }
}
