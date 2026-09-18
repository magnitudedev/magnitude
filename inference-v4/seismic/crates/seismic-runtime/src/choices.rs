//! Backend-owned execution choices. These are domains of currently implemented
//! forms, not hardware performance scores. All expansion is IR-only.
use crate::{execution::Execution, DeviceFacts};
use seismic_accounting::selection::{Choices, IntegerRange};
use seismic_lang::lowered_ir::LoweredIr;
use seismic_realization::{CallConv, Dispatch, LoadStrategy};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    CudaBlock { phase: usize },
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DispatchChoice;
impl Choices for DispatchChoice {
    type Alternative = Dispatch;
    fn len(&self) -> usize {
        2
    }
    fn get(&self, index: usize) -> Option<Dispatch> {
        [Dispatch::Sequential, Dispatch::ParallelRoot]
            .get(index)
            .copied()
    }
}

pub struct Domain {
    pub name: String,
    pub alternatives: seismic_accounting::selection::Domain,
}
pub enum Expansion {
    Choice(Domain),
    Execution(Execution),
    Infeasible(seismic_accounting::selection::CapacityViolation),
}
/// Decomposition parameters are fixed inputs here. Metal storage, reduction and
/// grouping choices remain dependent domains. Extending decomposition coverage
/// belongs to the compiler, not to a caller-supplied list of performance guesses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Form {
    CpuScalar,
    CudaScalar,
    #[cfg(target_os = "macos")]
    Metal {
        piece: Option<i64>,
        per_item: i64,
        split: i64,
        tile_piece: Option<i64>,
    },
}
pub struct Space<'a> {
    lowered: &'a LoweredIr,
    hardware: &'a DeviceFacts,
    form: Form,
}
impl<'a> Space<'a> {
    pub fn new(
        lowered: &'a LoweredIr,
        hardware: &'a DeviceFacts,
        form: Form,
    ) -> Result<Self, String> {
        let compatible = match (&form, hardware, lowered.backend.as_str()) {
            (Form::CpuScalar, DeviceFacts::Cpu { .. }, "cpu") => true,
            (Form::CudaScalar, DeviceFacts::Cuda(_), "cuda") => true,
            #[cfg(target_os = "macos")]
            (Form::Metal { .. }, DeviceFacts::Metal(_), "metal") => true,
            _ => false,
        };
        if !compatible {
            return Err("execution form, Lowered IR and hardware backend differ".into());
        }
        if let DeviceFacts::Cpu {
            architecture,
            operating_system,
        } = hardware
        {
            if *architecture != std::env::consts::ARCH || *operating_system != std::env::consts::OS
            {
                return Err(
                    "CPU scalar preparation currently targets the host architecture and ABI".into(),
                );
            }
        }
        Ok(Self {
            lowered,
            hardware,
            form,
        })
    }
    pub fn hardware(&self) -> &DeviceFacts {
        self.hardware
    }
    pub fn expand(&self, prefix: &[usize]) -> Result<Expansion, String> {
        let (function, consumed, _metal_loads) = match &self.form {
            Form::CpuScalar | Form::CudaScalar => {
                match seismic_lang::normalize::loads::expand(self.lowered, prefix)? {
                    seismic_lang::normalize::loads::Expansion::Choice(choice) => {
                        return Ok(Expansion::Choice(Domain {
                            name: format!(
                                "load site {} (variable {})",
                                choice.site, choice.variable
                            ),
                            alternatives: seismic_accounting::selection::Domain::new(choice)?,
                        }));
                    }
                    seismic_lang::normalize::loads::Expansion::Selected { function, consumed } => {
                        (function, consumed, LoadStrategy::Materialize)
                    }
                }
            }
            #[cfg(target_os = "macos")]
            Form::Metal { .. } => (self.lowered.clone(), 0, LoadStrategy::Materialize),
        };
        let prefix = &prefix[consumed..];
        match (&self.form, self.hardware) {
            (Form::CpuScalar, DeviceFacts::Cpu { .. }) => {
                if !prefix.is_empty() {
                    return Err("unused CPU execution decisions".into());
                }
                Ok(Expansion::Execution(Execution::Cpu(
                    seismic_cpu::prepare_resolved(&function)?,
                )))
            }
            (Form::CudaScalar, DeviceFacts::Cuda(facts)) => {
                let Some(&dispatch_index) = prefix.first() else {
                    return Ok(Expansion::Choice(Domain {
                        name: "work dispatch".into(),
                        alternatives: seismic_accounting::selection::Domain::new(DispatchChoice)?,
                    }));
                };
                let dispatch = DispatchChoice
                    .get(dispatch_index)
                    .ok_or("CUDA dispatch choice is outside its domain")?;
                let sequence = seismic_compiler::scalar_sequence_resolved(
                    &function,
                    CallConv::SystemV,
                    dispatch,
                )?;
                if sequence.phases.is_empty() {
                    return Err("CUDA form requires a nonempty phase sequence".into());
                }
                let limits = seismic_cuda::execution::Limits {
                    max_threads_per_block: facts.max_threads_per_block,
                    max_grid_x: facts.max_grid_x,
                };
                if limits.max_grid_x == 0 || limits.max_threads_per_block == 0 {
                    return Err("CUDA hardware requires positive dispatch capacities".into());
                }
                let mut selected = Vec::new();
                for (phase, program) in sequence.phases.into_iter().enumerate() {
                    let minimum = program
                        .program
                        .work_items
                        .div_ceil(u64::from(limits.max_grid_x))
                        .max(1);
                    let maximum = u64::from(limits.max_threads_per_block);
                    if minimum > maximum {
                        return Ok(Expansion::Infeasible(
                            seismic_accounting::selection::CapacityViolation {
                                resource: format!(
                                    "CUDA phase {phase} work items in one-dimensional grid"
                                ),
                                required: program.program.work_items,
                                available: u64::from(limits.max_grid_x) * maximum,
                            },
                        ));
                    }
                    let domain =
                        IntegerRange::new(Decision::CudaBlock { phase }, minimum, maximum)?;
                    let Some(&index) = prefix.get(phase + 1) else {
                        return Ok(Expansion::Choice(Domain {
                            name: format!("CUDA phase {phase} threads per block"),
                            alternatives: seismic_accounting::selection::Domain::new(domain)?,
                        }));
                    };
                    let threads = domain
                        .get(index)
                        .ok_or("CUDA block choice is outside its domain")?;
                    selected.push(seismic_cuda::execution::Execution::new(
                        program.program,
                        threads as u32,
                        limits,
                    )?);
                }
                if prefix.len() != selected.len() + 1 {
                    return Err("unused CUDA execution decisions".into());
                }
                Ok(Expansion::Execution(Execution::Cuda(selected)))
            }
            #[cfg(target_os = "macos")]
            (
                Form::Metal {
                    piece,
                    per_item,
                    split,
                    tile_piece,
                },
                DeviceFacts::Metal(facts),
            ) => {
                let config = seismic_metal::execution::Config {
                    loads: _metal_loads,
                    sg_per_tg: 1,
                    piece: *piece,
                    per_item: *per_item,
                    split: *split,
                    tile_piece: *tile_piece,
                    max_threads_per_threadgroup: facts
                        .max_threads_per_threadgroup
                        .try_into()
                        .map_err(|_| "Metal thread capacity exceeds integer range")?,
                    max_threadgroup_bytes: facts
                        .max_threadgroup_bytes
                        .try_into()
                        .map_err(|_| "Metal storage capacity exceeds integer range")?,
                };
                match seismic_metal::choices::expand(self.lowered, config, prefix)? {
                    seismic_metal::choices::Expansion::Infeasible {
                        launch,
                        required,
                        available,
                    } => Ok(Expansion::Infeasible(
                        seismic_accounting::selection::CapacityViolation {
                            resource: format!("Metal launch {launch} shared bytes per group"),
                            required,
                            available,
                        },
                    )),
                    seismic_metal::choices::Expansion::Choice(domain) => {
                        Ok(Expansion::Choice(Domain {
                            name: format!("Metal {:?}", domain.decision),
                            alternatives: seismic_accounting::selection::Domain::new(domain)?,
                        }))
                    }
                    seismic_metal::choices::Expansion::Execution {
                        execution,
                        consumed,
                    } => {
                        let end = consumed;
                        let family = seismic_metal::family::GroupFamily::derive(execution)?;
                        let mut groups = Vec::new();
                        for launch in 0..family.execution().memory().launches().len() {
                            let domain = family.grouping_choices(launch)?;
                            let Some(&index) = prefix.get(end + launch) else {
                                return Ok(Expansion::Choice(Domain {
                                    name: format!(
                                        "Metal launch {launch} work items per threadgroup"
                                    ),
                                    alternatives: seismic_accounting::selection::Domain::new(
                                        domain,
                                    )?,
                                }));
                            };
                            groups.push(
                                domain
                                    .get(index)
                                    .ok_or("Metal grouping choice is outside its domain")?,
                            );
                        }
                        if prefix.len() != end + groups.len() {
                            return Err("unused Metal execution decisions".into());
                        }
                        Ok(Expansion::Execution(Execution::Metal(
                            family.select_launches(&groups)?,
                        )))
                    }
                }
            }
            _ => unreachable!("execution form is checked at construction"),
        }
    }
}
