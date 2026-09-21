//! Engine-wide diagnostics. Engine-owned request facts (workload-envelope
//! requests, token/vocabulary policy, geometry validation) live here as
//! `Request`; runtime failures keep their typed shapes through
//! `From<seismic_runtime::Error>`. There is no string channel for
//! compiler or runtime structure.
use seismic_realization::failure::{ExecutionFailure, ExternalFailure, InvalidInvocation};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// A diagnostic of the caller's request: engine policy validation of
    /// workload envelopes, inputs, and model geometry.
    Request(String),
    /// An invocation rejected against its sealed contract.
    Invocation(InvalidInvocation),
    /// A synchronous execution failure (retained safety or external).
    Execution(ExecutionFailure),
    /// A charged allocation exceeds the device domain's remaining budget.
    Capacity { required: usize, available: usize },
    /// A memory limit below retained charges cannot be installed.
    LimitBelowCharges { limit: usize, charged: usize },
    /// A view or host transfer exceeds its bound byte range.
    Range { requested: usize, available: usize },
    /// The device, driver, allocation, or host system failed.
    External(ExternalFailure),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Request(message) => f.write_str(message),
            Self::Invocation(failure) => write!(f, "invalid invocation: {failure}"),
            Self::Execution(failure) => write!(f, "{failure}"),
            Self::Capacity {
                required,
                available,
            } => write!(
                f,
                "allocation requires {required} bytes; {available} charged bytes available"
            ),
            Self::LimitBelowCharges { limit, charged } => write!(
                f,
                "allocation limit {limit} cannot be below the {charged} retained charged bytes"
            ),
            Self::Range {
                requested,
                available,
            } => write!(f, "{requested} bytes requested of a {available}-byte bound range"),
            Self::External(failure) => write!(f, "{failure}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<&str> for Error {
    fn from(message: &str) -> Self {
        Self::Request(message.to_string())
    }
}

impl From<String> for Error {
    fn from(message: String) -> Self {
        Self::Request(message)
    }
}

impl From<seismic_runtime::Error> for Error {
    fn from(error: seismic_runtime::Error) -> Self {
        match error {
            seismic_runtime::Error::Capacity {
                required,
                available,
            } => Self::Capacity {
                required,
                available,
            },
            seismic_runtime::Error::LimitBelowCharges { limit, charged } => {
                Self::LimitBelowCharges { limit, charged }
            }
            seismic_runtime::Error::Range {
                requested,
                available,
            } => Self::Range {
                requested,
                available,
            },
            seismic_runtime::Error::External(failure) => Self::External(failure),
        }
    }
}
