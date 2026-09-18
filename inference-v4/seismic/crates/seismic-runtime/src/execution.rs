//! Backend executions prepared entirely before target emission and native compilation.
//! Preparation applies explicit choices; it does not assert their optimality.
use crate::{Candidate, DeviceFacts};
use seismic_lang::lowered_ir::LoweredIr;
use seismic_realization::ScalarProgram;

pub enum Execution {
    Cpu(ScalarProgram),
    Cuda(Vec<seismic_cuda::execution::Execution>),
    #[cfg(target_os = "macos")]
    Metal(seismic_metal::execution::Execution),
}
impl Execution {
    pub fn backend(&self) -> &'static str {
        match self {
            Self::Cpu(_) => "cpu",
            Self::Cuda(_) => "cuda",
            #[cfg(target_os = "macos")]
            Self::Metal(_) => "metal",
        }
    }
    pub fn prepare(
        lowered: &LoweredIr,
        candidate: Candidate,
        facts: &DeviceFacts,
    ) -> Result<Self, String> {
        let execution = match (candidate, facts) {
            (
                Candidate::Cpu { loads },
                DeviceFacts::Cpu {
                    architecture,
                    operating_system,
                },
            ) => {
                if *architecture != std::env::consts::ARCH
                    || *operating_system != std::env::consts::OS
                {
                    return Err(
                        "CPU execution preparation currently requires the host target".into(),
                    );
                }
                Self::Cpu(seismic_cpu::prepare(lowered, loads)?)
            }
            (
                Candidate::Cuda {
                    options,
                    threads_per_block,
                },
                DeviceFacts::Cuda(facts),
            ) => {
                if lowered.backend != "cuda" {
                    return Err("CUDA preparation requires CUDA Lowered IR".into());
                }
                let sequence = seismic_compiler::scalar_sequence(
                    lowered,
                    seismic_realization::CallConv::SystemV,
                    options,
                )?;
                if sequence.phases.is_empty() {
                    return Err("CUDA execution requires at least one phase".into());
                }
                let limits = seismic_cuda::execution::Limits {
                    max_threads_per_block: facts.max_threads_per_block,
                    max_grid_x: facts.max_grid_x,
                };
                Self::Cuda(
                    sequence
                        .phases
                        .into_iter()
                        .map(|phase| {
                            seismic_cuda::execution::Execution::new(
                                phase.program,
                                threads_per_block,
                                limits,
                            )
                        })
                        .collect::<Result<_, _>>()?,
                )
            }
            #[cfg(target_os = "macos")]
            (Candidate::Metal(mut config), DeviceFacts::Metal(facts)) => {
                if lowered.backend != "metal" {
                    return Err("Metal preparation requires Metal Lowered IR".into());
                }
                config.max_threads_per_threadgroup =
                    facts
                        .max_threads_per_threadgroup
                        .try_into()
                        .map_err(|_| "Metal thread capacity exceeds compiler integer range")?;
                config.max_threadgroup_bytes = facts
                    .max_threadgroup_bytes
                    .try_into()
                    .map_err(|_| "Metal shared-memory capacity exceeds compiler integer range")?;
                Self::Metal(seismic_metal::execution::prepare(lowered, config)?)
            }
            _ => return Err("execution choices and target hardware backend differ".into()),
        };
        Ok(execution)
    }

    /// Derive accounts from the same execution that emission consumes. These
    /// describe IR work/storage; they are not substitutes for native service models.
    pub fn account(&self) -> Result<Account, String> {
        Ok(match self {
            Self::Cpu(program) => {
                Account::Scalar(vec![seismic_accounting::realization::scalar(program)])
            }
            Self::Cuda(phases) => Account::Scalar(
                phases
                    .iter()
                    .map(|phase| seismic_accounting::realization::scalar(phase.program()))
                    .collect(),
            ),
            #[cfg(target_os = "macos")]
            Self::Metal(execution) => {
                let dispatches = execution
                    .phases()
                    .iter()
                    .flat_map(|phase| {
                        std::iter::once(phase.dispatch.clone()).chain(phase.merge_dispatch.clone())
                    })
                    .collect::<Vec<_>>();
                Account::Metal(seismic_accounting::storage::derive(
                    execution.memory(),
                    &dispatches,
                )?)
            }
        })
    }
}
pub enum Account {
    Scalar(Vec<seismic_accounting::realization::ScalarAccount>),
    #[cfg(target_os = "macos")]
    Metal(seismic_accounting::storage::StorageAccount),
}
