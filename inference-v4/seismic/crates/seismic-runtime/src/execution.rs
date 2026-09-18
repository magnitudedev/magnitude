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
        use seismic_accounting::selection::{Choices, IntegerRange};
        use seismic_compiler::tuner::Preparation;
        use seismic_cuda::tuning::{BlockChoice, DispatchChoice, FoldChoice};
        use seismic_lang::{ir::LoadMode, normalize::loads};
        use seismic_realization::LoadStrategy;

        let loads = match &candidate {
            Candidate::Cpu { loads } => *loads,
            Candidate::Cuda { options, .. } => options.loads,
            #[cfg(target_os = "macos")]
            Candidate::Metal(config) => config.loads,
        };
        let select = |domain: &seismic_accounting::selection::Domain| -> Result<usize, String> {
            if let Some(choice) = domain.owner::<loads::Choice>() {
                let mode = match loads {
                    LoadStrategy::Materialize => LoadMode::Materialize,
                    LoadStrategy::BorrowProvenReadOnly => LoadMode::Borrow,
                };
                return choice
                    .modes()
                    .iter()
                    .position(|&m| m == mode)
                    .ok_or_else(|| "diagnostic load choice is outside its derived domain".into());
            }
            if domain.owner::<FoldChoice>().is_some() { return Ok(0); }
            if let Some(choice) = domain.owner::<DispatchChoice>() {
                let Candidate::Cuda { options, .. } = &candidate else {
                    return Err("CUDA dispatch choice for a different backend".into());
                };
                return (0..choice.len())
                    .find(|&i| choice.get(i) == Some(options.dispatch))
                    .ok_or_else(|| "diagnostic dispatch is outside its derived domain".into());
            }
            if let Some(choice) = domain.owner::<IntegerRange<BlockChoice>>() {
                let Candidate::Cuda {
                    threads_per_block, ..
                } = &candidate
                else {
                    return Err("CUDA block choice for a different backend".into());
                };
                let lanes = choice
                    .decision
                    .family()
                    .lanes_per_item(choice.decision.phase)
                    .ok_or("missing CUDA phase participant geometry")?;
                if !threads_per_block.is_multiple_of(lanes) {
                    return Err(
                        "diagnostic CUDA threads do not contain whole work-item participant groups"
                            .into(),
                    );
                }
                return choice
                    .index(u64::from(*threads_per_block / lanes))
                    .ok_or_else(|| {
                        format!(
                            "threads per block is outside CUDA phase {}'s derived domain",
                            choice.decision.phase
                        )
                    });
            }
            #[cfg(target_os = "macos")]
            if let Some(choice) = domain.owner::<seismic_metal::choices::Domain>() {
                let Candidate::Metal(_) = &candidate else {
                    return Err("Metal implementation choice for a different backend".into());
                };
                let selected = choice.diagnostic(loads);
                return choice
                    .index(&selected)
                    .ok_or_else(|| "diagnostic Metal choice is outside its derived domain".into());
            }
            #[cfg(target_os = "macos")]
            if let Some(choice) = domain.owner::<seismic_metal::family::GroupingChoices>() {
                let Candidate::Metal(config) = &candidate else {
                    return Err("Metal grouping choice for a different backend".into());
                };
                let groups =
                    u64::try_from(config.sg_per_tg).map_err(|_| "negative Metal grouping")?;
                return choice.index(groups).ok_or_else(|| {
                    format!(
                        "work items per threadgroup is outside Metal launch {}'s derived domain",
                        choice.launch
                    )
                });
            }
            Err("unhandled diagnostic implementation choice".into())
        };
        let prepare = |path: &[usize]| -> Result<Preparation<Execution>, String> {
            use crate::tuner::map_preparation;
            Ok(match (&candidate, facts) {
                (
                    Candidate::Cpu { .. },
                    DeviceFacts::Cpu {
                        architecture,
                        operating_system,
                    },
                ) if *architecture == std::env::consts::ARCH
                    && *operating_system == std::env::consts::OS =>
                {
                    map_preparation(seismic_cpu::tuning::prepare(lowered, path)?, Execution::Cpu)
                }
                (Candidate::Cuda { .. }, DeviceFacts::Cuda(device)) => map_preparation(
                    seismic_cuda::tuning::prepare(lowered, device, path)?,
                    Execution::Cuda,
                ),
                #[cfg(target_os = "macos")]
                (Candidate::Metal(config), DeviceFacts::Metal(device)) => {
                    let form =
                        seismic_metal::tuning::Form::Fixed(seismic_metal::tuning::Decomposition {
                            per_item: config.per_item,
                            split: config.split,
                            tile_piece: config.tile_piece,
                        });
                    let capacities = seismic_metal::tuning::Capacities::from_device(device);
                    map_preparation(
                        seismic_metal::tuning::expand(lowered, &form, &capacities, path)?,
                        Execution::Metal,
                    )
                }
                _ => return Err("explicit candidate and device backend differ".into()),
            })
        };
        let mut prefix = Vec::new();
        let mut prepared = prepare(&prefix)?;
        loop {
            match prepared {
                Preparation::Choice { alternatives, .. } => {
                    let index = select(&alternatives)?;
                    prefix.push(index);
                    #[cfg(target_os = "macos")]
                    if let Some(choice) = alternatives.owner::<seismic_metal::family::GroupingChoices>() {
                        prepared = crate::tuner::map_preparation(choice.refine(index)?, Execution::Metal);
                        continue;
                    }
                    prepared = prepare(&prefix)?;
                }
                Preparation::Execution(execution) => return Ok(execution),
                Preparation::Infeasible(violation) => {
                    return Err(format!(
                        "{} requires {} units, capacity is {}",
                        violation.resource, violation.required, violation.available,
                    ));
                }
            }
        }
    }

    /// Read-only projections of this exact progressively refined execution.
    /// They describe the named representation and never supply timings, proofs,
    /// or observations. CUDA's scalar account deliberately remains attached to
    /// its pre-PTX representation; terminal operation structure comes from PTX.
    pub fn account(&self) -> Result<Account<'_>, String> {
        Ok(match self {
            Self::Cpu(program) => Account::CpuScalarIr {
                implementation: program,
                account: seismic_accounting::realization::scalar(program),
            },
            Self::Cuda(phases) => Account::Cuda(
                phases
                    .iter()
                    .map(|execution| CudaPhaseAccount {
                        execution,
                        scalar_ir: seismic_accounting::realization::scalar(execution.program()),
                    })
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
                Account::MetalStorage {
                    implementation: execution,
                    account: seismic_metal::model::derive(execution.memory(), &dispatches)?,
                }
            }
        })
    }
}

pub enum Account<'a> {
    CpuScalarIr {
        implementation: &'a ScalarProgram,
        account: seismic_accounting::realization::ScalarAccount,
    },
    Cuda(Vec<CudaPhaseAccount<'a>>),
    #[cfg(target_os = "macos")]
    MetalStorage {
        implementation: &'a seismic_metal::execution::Execution,
        account: seismic_metal::model::StorageAccount,
    },
}
/// Both projections belong to the same selected CUDA phase. The scalar count
/// excludes PTX-introduced guards, address instructions and helper expansion;
/// those instructions are available only from the retained target implementation.
/// Runtime prediction additionally requires hardware and invocation inputs.
pub struct CudaPhaseAccount<'a> {
    execution: &'a seismic_cuda::execution::Execution,
    scalar_ir: seismic_accounting::realization::ScalarAccount,
}
impl<'a> CudaPhaseAccount<'a> {
    pub fn implementation(&self) -> &'a seismic_cuda::execution::Execution {
        self.execution
    }
    pub fn scalar_ir(&self) -> &seismic_accounting::realization::ScalarAccount {
        &self.scalar_ir
    }
    pub fn target_ir(&self) -> &'a seismic_cuda::ptx::TargetPlan {
        self.execution.target_plan()
    }
}
