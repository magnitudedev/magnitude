//! Common legalization: the portable matrix and safety
//! obligation consumption.

pub mod matrix;
pub mod safety;

pub use matrix::{
    registry_math_is_versioned, universal_form, universal_numerical, DataAccess, LayoutTransform,
    LegalizationBug, LinearLoopOp, SeismicMathReference, UniversalForm, UniversalLegalization,
    SEISMIC_MATH, SEISMIC_MATH_IDENTITY, SEISMIC_MATH_VERSION,
};
pub use safety::{
    discharge, discharge_precondition, CheckPredicate, GraphFacts, InactiveBehavior,
    ObligationDischarge, RuntimeCheck, SafetyKind, StaticProof, StatusWrite,
};
