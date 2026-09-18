//! IR-only selection boundary. A backend space couples legality, execution
//! construction and resource derivation; native compilation consumes its result.
use crate::{execution::Execution, DeviceFacts};
use seismic_accounting::selection::{self, Certificate, Cost, Selected, Space};

/// Implemented by a backend's complete execution form, not by native compilation
/// or benchmark-driven candidate generation. Hardware facts are immutable inputs.
pub trait BackendSpace: Space<Execution = Execution> {
    fn hardware(&self) -> &DeviceFacts;
}

pub struct TunedIr {
    selected: Selected<Execution>,
    hardware: DeviceFacts,
}
impl TunedIr {
    pub fn execution(&self) -> &Execution {
        self.selected.execution()
    }
    pub fn certificate(&self) -> &Certificate {
        self.selected.certificate()
    }
    pub fn modeled_cost(&self) -> Cost {
        self.selected.cost()
    }
    pub fn hardware(&self) -> &DeviceFacts {
        &self.hardware
    }
    pub(crate) fn into_execution(self) -> Execution {
        self.selected.into_execution()
    }
}

pub fn tune<S: BackendSpace>(space: &S, node_budget: usize) -> Result<TunedIr, String> {
    let selected = selection::select(space, node_budget)?;
    let backend = match space.hardware() {
        DeviceFacts::Cpu { .. } => "cpu",
        DeviceFacts::Cuda(_) => "cuda",
        #[cfg(target_os = "macos")]
        DeviceFacts::Metal(_) => "metal",
    };
    if selected.execution().backend() != backend {
        return Err("execution space produced an execution for a different backend".into());
    }
    Ok(TunedIr {
        selected,
        hardware: space.hardware().clone(),
    })
}

pub fn verify<S: BackendSpace>(
    space: &S,
    tuned: &TunedIr,
    node_budget: usize,
) -> Result<(), String> {
    if space.hardware() != tuned.hardware() {
        return Err("tuned execution hardware assumptions changed".into());
    }
    let cost = selection::verify(space, tuned.certificate(), node_budget)?;
    if cost != tuned.modeled_cost() {
        return Err("tuned execution objective differs from its certificate".into());
    }
    Ok(())
}
