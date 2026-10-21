//! The strict tensor binder: every weight a definition names is bound by its
//! exact stored name and shape, and at the end every stored tensor must have
//! been bound.

use crate::HeaderError;
use magnitude_artifacts::gguf::{Directory, Encoding, TensorDescriptor};
use magnitude_family_contracts::{
    checked_product, ImportTransform, ScaledActivations, WeightDescriptor,
};
use std::collections::HashSet;

pub struct Tensors<'a> {
    directory: &'a Directory,
    bound: HashSet<&'a str>,
}

impl<'a> Tensors<'a> {
    pub fn new(directory: &'a Directory) -> Self {
        Self {
            directory,
            bound: HashSet::new(),
        }
    }

    fn stored(&self, name: &str) -> Result<&'a TensorDescriptor, HeaderError> {
        self.directory
            .tensor(name)
            .ok_or_else(|| HeaderError::MissingWeight(name.to_owned()))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.directory.tensor(name).is_some()
    }

    /// The stored shape of `name`, outermost first.
    pub fn shape(&self, name: &str) -> Result<&'a [u64], HeaderError> {
        Ok(&self.stored(name)?.shape)
    }

    /// The outermost extent of a stored matrix.
    pub fn rows(&self, name: &str) -> Result<u64, HeaderError> {
        match self.shape(name)? {
            [rows, _] => Ok(*rows),
            shape => Err(HeaderError::NotMatrix {
                name: name.to_owned(),
                shape: shape.to_vec(),
            }),
        }
    }

    /// Binds `name` as stored; its stored shape must be `shape`.
    pub fn bind(&mut self, name: &str, shape: &[u64]) -> Result<WeightDescriptor, HeaderError> {
        self.bind_transformed(name, shape, Vec::new())
    }

    /// Accepts `name`, when stored, as a weight the model never reads: its
    /// stored shape must be `shape`, it counts as bound, and no descriptor
    /// names it, so it is never planned or imported.
    pub fn bind_unused(&mut self, name: &str, shape: &[u64]) -> Result<(), HeaderError> {
        if self.contains(name) {
            self.bind(name, shape)?;
        }
        Ok(())
    }

    /// Binds `name`, stored exactly as `stored`, through exact import
    /// `transforms`; the descriptor's shape is the one they produce.
    pub fn bind_transformed(
        &mut self,
        name: &str,
        stored: &[u64],
        transforms: Vec<ImportTransform>,
    ) -> Result<WeightDescriptor, HeaderError> {
        let tensor = self.stored(name)?;
        if tensor.shape != stored {
            return Err(HeaderError::WeightShape {
                name: name.to_owned(),
                expected: stored.to_vec(),
                received: tensor.shape.clone(),
            });
        }
        checked_product(stored)?;
        let mut descriptor = WeightDescriptor {
            name: name.to_owned(),
            shape: stored.to_vec(),
            transforms,
        };
        descriptor.shape = descriptor.transformed_shape(stored)?;
        self.bound.insert(tensor.name.as_str());
        Ok(descriptor)
    }

    /// Binds `name` as stored, scaled by its stored second-level scale (see
    /// [`Self::companion_scale`]) when the file has one: `shape`'s outermost
    /// extent is the matrix count of a stacked (rank 3) weight, else one.
    pub fn bind_scaled(
        &mut self,
        name: &str,
        shape: &[u64],
    ) -> Result<WeightDescriptor, HeaderError> {
        let mut weight = self.bind(name, shape)?;
        let matrices = match shape {
            [stack, _, _] => *stack,
            _ => 1,
        };
        if let Some(scale) = self.companion_scale(name, matrices)? {
            weight.transforms.push(scale);
        }
        Ok(weight)
    }

    /// Binds the stored F32 `<stem>.scale` companion of `<stem>.weight`
    /// (NVFP4's per-tensor or per-matrix second-level scale), one value per
    /// matrix, as the weight's `ScaleByTensor`; `None` when the file stores
    /// none.
    ///
    /// A scaled weight whose producer calibrated it for FP4 activations
    /// (W4A4) also stores `<stem>.input_scale`, its static activation scale:
    /// its products take NVFP4 operands. The operand recipe scales each row
    /// by its own power of two, so the stored value is checked and
    /// acknowledged, not applied.
    pub fn companion_scale(
        &mut self,
        weight: &str,
        matrices: u64,
    ) -> Result<Option<ImportTransform>, HeaderError> {
        let Some(stem) = weight.strip_suffix(".weight") else {
            return Ok(None);
        };
        let Some(scale) = self.directory.tensor(&format!("{stem}.scale")) else {
            return Ok(None);
        };
        let input = self.directory.tensor(&format!("{stem}.input_scale"));
        for companion in std::iter::once(scale).chain(input) {
            if companion.encoding != Encoding::F32 || companion.shape != [matrices] {
                return Err(HeaderError::CompanionScale {
                    name: companion.name.clone(),
                    matrices,
                    encoding: companion.encoding,
                    shape: companion.shape.clone(),
                });
            }
            self.bound.insert(companion.name.as_str());
        }
        Ok(Some(ImportTransform::ScaleByTensor {
            tensor: scale.name.clone(),
            activations: match input {
                Some(_) => ScaledActivations::Nvfp4,
                None => ScaledActivations::Rounded,
            },
        }))
    }

    /// The stored tensors bound by no role, in directory order.
    pub fn unbound(&self) -> impl Iterator<Item = &'a str> + '_ {
        self.directory
            .tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .filter(|name| !self.bound.contains(name))
    }

    /// Proves every stored tensor was bound, naming every one that was not.
    pub fn finish(self) -> Result<(), HeaderError> {
        let unbound = self.unbound().map(str::to_owned).collect::<Vec<_>>();
        if unbound.is_empty() {
            Ok(())
        } else {
            Err(HeaderError::UnboundWeights(unbound))
        }
    }
}
