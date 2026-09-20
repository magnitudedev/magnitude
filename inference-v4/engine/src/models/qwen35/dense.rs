//! Retained execution resources for the dense feedforward suffix of a Qwen block.
use crate::weights::residency::ResidentWeight;
use seismic_lang::{
    sir::Program,
    types::{DType, Elem},
};
use seismic_runtime::{
    plan::{Bindings, CompiledPlan, PlanCompiler, Settings},
    Buffer, Device,
};
use std::collections::HashMap;

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
/// Owns native code and weights. The compiler/runtime own all result destinations.
pub struct DenseSuffix {
    plan: CompiledPlan,
    weights: DenseWeights,
    epsilon: f32,
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
        let shape = |n: u64| {
            i64::try_from(n).map_err(|_| "dense suffix dimension exceeds index range".to_string())
        };
        let shapes = HashMap::from([
            (
                "M".into(),
                i64::try_from(rows).map_err(|_| "row count exceeds index range")?,
            ),
            ("H".into(), shape(*hidden)?),
            ("F".into(), shape(*intermediate)?),
        ]);
        let elements = HashMap::from([
            ("A".into(), Elem::Dtype(activation)),
            ("NW".into(), weights.norm.element().clone()),
            ("GW".into(), weights.gate.element().clone()),
            ("UW".into(), weights.up.element().clone()),
            ("DW".into(), weights.down.element().clone()),
        ]);
        let plan = PlanCompiler::new(device, program, settings).compile_entry(
            "qwen_dense_suffix",
            &shapes,
            &elements,
        )?;
        Ok(Self {
            plan,
            weights,
            epsilon,
        })
    }
    pub fn execute(&mut self, residual: &Buffer) -> Result<Buffer, String> {
        let results = self.plan.execute_with_results(&Invocation {
            weights: &self.weights,
            epsilon: self.epsilon,
            residual,
        })?;
        results
            .into_iter()
            .find(|result| result.path == [6] && result.plane.is_empty())
            .map(|result| result.buffer)
            .ok_or_else(|| "dense suffix returned no F32 residual result".into())
    }
}
struct Invocation<'a> {
    weights: &'a DenseWeights,
    epsilon: f32,
    residual: &'a Buffer,
}
impl Bindings for Invocation<'_> {
    fn buffer(&self, root: &str, plane: &str) -> Option<&Buffer> {
        let weight = match root {
            "norm" => Some(&self.weights.norm),
            "gate_weight" => Some(&self.weights.gate),
            "up_weight" => Some(&self.weights.up),
            "down_weight" => Some(&self.weights.down),
            _ => None,
        };
        if let Some(weight) = weight {
            return weight.plane(plane);
        }
        if !plane.is_empty() {
            return None;
        }
        match root {
            "residual" => Some(self.residual),
            _ => None,
        }
    }
    fn scalar(&self, name: &str) -> Option<f64> {
        (name == "eps").then_some(f64::from(self.epsilon))
    }
}
