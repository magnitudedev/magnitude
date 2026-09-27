//! The strict tensor binder: every weight a definition names is bound by its
//! exact stored name and shape, and at the end every stored tensor must have
//! been bound.

use crate::HeaderError;
use magnitude_artifacts::gguf::{Directory, Encoding, TensorDescriptor};
use magnitude_family_contracts::{checked_product, ImportTransform, WeightDescriptor};
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
            weight
                .transforms
                .push(ImportTransform::ScaleByTensor { tensor: scale });
        }
        Ok(weight)
    }

    /// Binds the stored F32 `<stem>.scale` companion of `<stem>.weight`
    /// (NVFP4's per-tensor or per-matrix second-level scale), one value per
    /// matrix, and names it; `None` when the file stores none.
    pub fn companion_scale(
        &mut self,
        weight: &str,
        matrices: u64,
    ) -> Result<Option<String>, HeaderError> {
        let Some(scale) = weight
            .strip_suffix(".weight")
            .and_then(|stem| self.directory.tensor(&format!("{stem}.scale")))
        else {
            return Ok(None);
        };
        if scale.encoding != Encoding::F32 || scale.shape != [matrices] {
            return Err(HeaderError::CompanionScale {
                name: scale.name.clone(),
                matrices,
                encoding: scale.encoding,
                shape: scale.shape.clone(),
            });
        }
        self.bound.insert(scale.name.as_str());
        Ok(Some(scale.name.clone()))
    }

    /// The stored tensors bound by no role, in directory order.
    pub fn unbound(&self) -> impl Iterator<Item = &'a str> + '_ {
        self.directory
            .tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .filter(|name| !self.bound.contains(name))
    }

    /// Proves every stored tensor was bound.
    pub fn finish(self) -> Result<(), HeaderError> {
        match self.unbound().next() {
            Some(name) => Err(HeaderError::UnboundWeight(name.to_owned())),
            None => Ok(()),
        }
    }
}
