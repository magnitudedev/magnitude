//! Prepared dense feedforward suffix of a Qwen block.

use crate::{kernels, weights::residency::ResidentWeight};
use seismic::{DType, Device, Element, Kernel, PrecisionPolicy, Tensor};

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

pub struct DenseSuffix {
    kernel: Kernel<kernels::qwen_dense_suffix::Entry>,
    weights: DenseWeights,
    epsilon: f32,
    rows: u64,
}

impl DenseSuffix {
    pub fn compile(
        device: &Device,
        invocation: DenseInvocation,
        weights: DenseWeights,
        precision: PrecisionPolicy,
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
            || [&weights.norm, &weights.gate, &weights.up, &weights.down]
                .into_iter()
                .any(|weight| !weight.belongs_to(device))
        {
            return Err("dense suffix weights disagree on device or dimensions".into());
        }
        let rows = u64::try_from(rows)
            .map_err(|_| "dense suffix row count exceeds the Seismic shape domain")?;
        let kernel = kernels::qwen_dense_suffix::for_device_with(
            device,
            precision,
            kernels::qwen_dense_suffix::Elements {
                A: Element::dense(activation),
                NW: weights.norm.element(),
                GW: weights.gate.element(),
                UW: weights.up.element(),
                DW: weights.down.element(),
            },
        )
        .map_err(|error| error.to_string())?;
        Ok(Self {
            kernel,
            weights,
            epsilon,
            rows,
        })
    }

    pub fn kernel_count(&self) -> usize {
        1
    }

    pub fn execute(&self, residual: &Tensor) -> Result<Tensor, String> {
        if residual.extents() != [self.rows, self.weights.norm.descriptor().shape[0]] {
            return Err("dense suffix residual has the wrong shape".into());
        }
        self.kernel
            .call(kernels::qwen_dense_suffix::Args {
                residual,
                norm: self.weights.norm.tensor(),
                gate_weight: self.weights.gate.tensor(),
                up_weight: self.weights.up.tensor(),
                down_weight: self.weights.down.tensor(),
                eps: self.epsilon,
            })
            .map(|results| results.r6)
            .map_err(|error| error.to_string())
    }

    pub fn weights(&self) -> &DenseWeights {
        &self.weights
    }
}
