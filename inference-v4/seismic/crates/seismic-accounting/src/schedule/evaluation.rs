//! Original execution constraints, independent of selection strategy.
//! Both representations append to a caller-owned mathematical model through
//! `schedule::export`; neither owns a search coordinator.
use super::{structured, *};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Model {
    Flat(super::Model),
    Structured {
        model: structured::Structured,
        expansion_limit: u64,
    },
}
impl From<super::Model> for Model {
    fn from(model: super::Model) -> Self {
        Self::Flat(model)
    }
}
impl Model {
    pub(crate) fn lower_bound(&self) -> Result<u64, String> {
        match self {
            Self::Flat(model) => model.lower_bound(),
            Self::Structured { model, .. } => model.lower_bound(),
        }
    }
    pub fn into_flat(self) -> Result<super::Model, String> {
        match self {
            Self::Flat(model) => Ok(model),
            Self::Structured { .. } => Err("execution model retains structured work".into()),
        }
    }
    pub fn timebase(&self) -> &Timebase {
        match self {
            Self::Flat(m) => &m.timebase,
            Self::Structured { model, .. } => &model.timebase,
        }
    }
    pub fn relationship(&self) -> &crate::authority::ModelRelationship {
        match self {
            Self::Flat(m) => &m.relationship,
            Self::Structured { model, .. } => &model.relationship,
        }
    }
}
