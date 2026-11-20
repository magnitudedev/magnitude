//! Shared dialect-perturbation helpers for family tests.
//!
//! A GGUF dialect variation (omitting a defaulted key, adding a benign key
//! with its invariant value, or reshaping singleton axes) must produce an
//! identical model definition; mutating an architectural invariant must be
//! rejected.

use magnitude_artifacts::gguf::{Directory, Metadata, Value};
use std::fmt::Debug;

/// Removes metadata entry `key` from `directory`, if present.
pub fn remove_metadata(directory: &mut Directory, key: &str) {
    directory.metadata.retain(|entry| entry.name != key);
}

/// Sets metadata entry `key` to `value` in `directory`, replacing any existing
/// entry with that name.
pub fn set_metadata(directory: &mut Directory, key: &str, value: Value) {
    directory.metadata.retain(|entry| entry.name != key);
    directory.metadata.push(Metadata {
        name: key.into(),
        value,
    });
}

/// Replaces the recorded shape of tensor `name` in `directory`.
pub fn reshape_tensor(directory: &mut Directory, name: &str, shape: &[u64]) {
    directory
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == name)
        .unwrap_or_else(|| panic!("tensor {name:?} is recorded"))
        .shape = shape.to_vec();
}

/// Asserts that applying `perturb` to `base` produces an identical output from
/// `inspect`.
pub fn assert_dialect_equivalent<T, E>(
    base: &Directory,
    inspect: impl Fn(&Directory) -> Result<T, E>,
    perturb: impl FnOnce(&mut Directory),
) where
    T: PartialEq + Debug,
    E: Debug,
{
    let expected = inspect(base).expect("base directory must be admitted");
    let mut perturbed = base.clone();
    perturb(&mut perturbed);
    let actual = inspect(&perturbed).expect("dialect-perturbed directory must be admitted");
    assert_eq!(actual, expected, "dialect perturbation changed the definition");
}

/// Asserts that applying `perturb` to `base` is rejected by `inspect`,
/// returning the error for further inspection.
pub fn assert_dialect_rejected<T, E>(
    base: &Directory,
    inspect: impl Fn(&Directory) -> Result<T, E>,
    perturb: impl FnOnce(&mut Directory),
) -> E
where
    T: Debug,
    E: Debug,
{
    let mut perturbed = base.clone();
    perturb(&mut perturbed);
    inspect(&perturbed).expect_err("invariant-violating perturbation must be rejected")
}
