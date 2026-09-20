//! CPU target configuration for constructive physical compilation.

mod estimate;
pub use estimate::{EstimateModel, IDENTITY, Totals};

pub const TARGET: &str = "cpu";
pub const SCRATCH_BYTES: u64 = 64 << 20;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Limits {
    pub workers: u64,
    pub max_scratch_bytes: u64,
}

impl Limits {
    pub fn host(workers: u64) -> Self {
        Self {
            workers,
            max_scratch_bytes: SCRATCH_BYTES,
        }
    }
}

pub struct Cpu {
    pub(crate) limits: Limits,
    pub(crate) executable_target: seismic_realization::executable::ExecutableTargetProfile<
        seismic_compiler::terminal::ScalarCapabilitySet,
    >,
}

impl Cpu {
    pub fn new(limits: Limits, estimate: EstimateModel) -> Result<Self, String> {
        if limits.workers == 0 || i64::try_from(limits.max_scratch_bytes).is_err() {
            return Err(format!(
                "CPU compilation needs at least one worker and representable scratch; received {} workers and {} bytes",
                limits.workers, limits.max_scratch_bytes
            ));
        }
        estimate.validate()?;
        let executable_target = crate::physical::target_profile(&limits);
        Ok(Self {
            limits,
            executable_target,
        })
    }

    pub fn host(workers: u64) -> Result<Self, String> {
        Self::new(Limits::host(workers), EstimateModel::unqualified(workers))
    }

    pub fn limits(&self) -> &Limits {
        &self.limits
    }
}
