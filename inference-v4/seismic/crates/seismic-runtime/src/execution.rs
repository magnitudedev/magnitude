//! Backend executions prepared entirely before target emission and native compilation.
//! Only the tuner can authorize native compilation of these internal executions.
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
}
