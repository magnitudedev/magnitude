//! Structured runtime failures preserve resource facts through preparation.
//!
//! Every failure carries its typed fact: the memory domain reports its
//! remaining budget, view/transfer bounds report the requested and available
//! extents, and backend device/driver/host failures are the shared
//! `ExternalFailure` taxonomy. No variant is a bare string channel.
use seismic_realization::failure::{ExternalFailure, ExternalStage};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// A charged allocation exceeds the domain's remaining budget.
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
            Self::Range { requested, available } => write!(
                f,
                "{requested} bytes requested of a {available}-byte bound range"
            ),
            Self::External(failure) => write!(f, "{failure}"),
        }
    }
}

impl std::error::Error for Error {}

pub(crate) fn external(stage: ExternalStage, detail: impl Into<String>) -> Error {
    Error::External(ExternalFailure {
        stage,
        detail: detail.into(),
    })
}
