//! Sound factor relaxations are derived from the typed mathematical model.
use crate::model::{Assessment, Domain, Factor, Result};
pub fn assess(factor: &Factor, domains: &[Domain]) -> Result<Assessment> {
    factor.assess(domains)
}
