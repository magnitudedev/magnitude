//! Engine-owned request failures plus the typed public Seismic failures.
//!
//! Compiler/runtime contradictions are panics at their private owner and do
//! not enter this enum. Conversely, every Seismic condition a model consumer
//! can actually encounter retains its public typed shape here.

#[derive(Debug)]
pub enum Error {
    /// Invalid model metadata, engine policy, or caller-owned request data.
    Request(String),
    Target(seismic::TargetError),
    Load(seismic::LoadError),
    Call(seismic::CallError),
    Tensor(seismic::TensorError),
    MemoryLimit(seismic::MemoryLimitError),
    Execution(seismic::ExecutionError),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Request(message) => f.write_str(message),
            Self::Target(error) => write!(f, "{error}"),
            Self::Load(error) => write!(f, "{error}"),
            Self::Call(error) => write!(f, "{error}"),
            Self::Tensor(error) => write!(f, "{error}"),
            Self::MemoryLimit(error) => write!(f, "{error}"),
            Self::Execution(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<&str> for Error {
    fn from(message: &str) -> Self {
        Self::Request(message.to_owned())
    }
}

impl From<String> for Error {
    fn from(message: String) -> Self {
        Self::Request(message)
    }
}

macro_rules! typed_error {
    ($source:ty, $variant:ident) => {
        impl From<$source> for Error {
            fn from(error: $source) -> Self {
                Self::$variant(error)
            }
        }
    };
}

typed_error!(seismic::TargetError, Target);
typed_error!(seismic::LoadError, Load);
typed_error!(seismic::CallError, Call);
impl From<seismic::WorkflowError> for Error {
    fn from(error: seismic::WorkflowError) -> Self {
        Self::Call(seismic::CallError::Workflow(error))
    }
}
typed_error!(seismic::TensorError, Tensor);
typed_error!(seismic::MemoryLimitError, MemoryLimit);
typed_error!(seismic::ExecutionError, Execution);
