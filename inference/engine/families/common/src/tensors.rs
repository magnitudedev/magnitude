//! The strict tensor binder: every weight a definition names is bound by its
//! exact stored name and shape, and at the end every stored tensor must have
//! been bound.

use crate::{CompanionForm, HeaderError};
use magnitude_artifacts::gguf::{Directory, Encoding, TensorDescriptor};
use magnitude_family_contracts::{
    checked_product, ImportTransform, ScaleCodes, ScaledActivations, WeightDescriptor,
};
use std::collections::HashSet;

/// The metadata key declaring the codes of a file's I8 block-scale grids
/// (`Tensors::companion_scale`).
pub const SCALE_ENCODING: &str = "magnitude.scale_encoding";
/// Its one admitted value: UE8M0 bytes (`ScaleCodes::Ue8m0`).
pub const UE8M0: &str = "ue8m0";

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

    /// Binds `name` as stored, scaled by its stored companion scale (see
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

    /// Binds the stored companion `<stem>.scale` of `<stem>.weight` as the
    /// weight's scale: of an F8_E4M3 matrix, which takes one, its grid of
    /// UE8M0 block scales ([`Self::scale_grid`]); of any other weight,
    /// NVFP4's per-tensor or per-matrix second-level scale, one F32 value
    /// per matrix, as its `ScaleByTensor`. `None` when the file stores none
    /// for a weight that takes none.
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
        let stored = self.stored(weight)?;
        let name = format!("{stem}.scale");
        let scale = self.directory.tensor(&name);
        if stored.encoding == Encoding::F8E4M3 {
            let scale = scale.ok_or(HeaderError::MissingWeight(name))?;
            return self.scale_grid(stored, scale).map(Some);
        }
        let Some(scale) = scale else {
            return Ok(None);
        };
        let input = self.directory.tensor(&format!("{stem}.input_scale"));
        for companion in std::iter::once(scale).chain(input) {
            if companion.encoding != Encoding::F32 || companion.shape != [matrices] {
                return Err(HeaderError::CompanionScale {
                    name: companion.name.clone(),
                    expected: CompanionForm::PerMatrix { matrices },
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

    /// Binds `scale` as the grid of UE8M0 block scales of the F8_E4M3
    /// matrix `weight` (`ImportTransform::ScaleBlocks`): I8 `[rows / r,
    /// columns / c]` for the tiles `[r, c]` it tiles the matrix by
    /// (`ImportTransform::grid_tile`; DeepSeek's 128×128, a table's 1×32),
    /// in a file declaring [`SCALE_ENCODING`] =
    /// [`UE8M0`], since an I8 tensor's bytes are codes only by that
    /// declaration. Any other scale is refused, naming it.
    fn scale_grid(
        &mut self,
        weight: &TensorDescriptor,
        scale: &'a TensorDescriptor,
    ) -> Result<ImportTransform, HeaderError> {
        let &[rows, columns] = &weight.shape[..] else {
            return Err(HeaderError::NotMatrix {
                name: weight.name.clone(),
                shape: weight.shape.clone(),
            });
        };
        let declared =
            self.directory.value(SCALE_ENCODING).and_then(|value| value.string()) == Some(UE8M0);
        let tile = ImportTransform::grid_tile(&weight.shape, &scale.shape)
            .filter(|_| declared && scale.encoding == Encoding::I8)
            .ok_or_else(|| HeaderError::CompanionScale {
                name: scale.name.clone(),
                expected: CompanionForm::Ue8m0Grid { rows, columns },
                encoding: scale.encoding,
                shape: scale.shape.clone(),
            })?;
        self.bound.insert(scale.name.as_str());
        Ok(ImportTransform::ScaleBlocks {
            tensor: scale.name.clone(),
            tile,
            codes: ScaleCodes::Ue8m0,
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_artifacts::gguf::{ByteOrder, Metadata, Scalar, Value};

    const WEIGHT: &str = "blk.0.attn_q_b.weight";
    const SCALE: &str = "blk.0.attn_q_b.scale";

    const SHAPE: [u64; 2] = [64, 256];

    /// A file of an F8_E4M3 [`SHAPE`] weight and, when `scale` gives it,
    /// its `.scale`, declaring `declared` as its scale encoding.
    fn directory(declared: Option<&str>, scale: Option<(Encoding, Vec<u64>)>) -> Directory {
        shaped(SHAPE, declared, scale)
    }

    fn shaped(weight: [u64; 2], declared: Option<&str>, scale: Option<(Encoding, Vec<u64>)>) -> Directory {
        let tensor = |name: &str, shape: Vec<u64>, encoding| TensorDescriptor {
            name: name.into(),
            shape,
            encoding,
            offset: 0,
            nbytes: 0,
        };
        Directory {
            version: 3,
            byte_order: ByteOrder::Little,
            alignment: 32,
            data_offset: 0,
            metadata: declared
                .map(|value| Metadata {
                    name: SCALE_ENCODING.into(),
                    value: Value::Scalar(Scalar::String(value.into())),
                })
                .into_iter()
                .collect(),
            tensors: std::iter::once(tensor(WEIGHT, weight.to_vec(), Encoding::F8E4M3))
                .chain(scale.map(|(encoding, shape)| tensor(SCALE, shape, encoding)))
                .collect(),
        }
    }

    #[test]
    fn a_declared_ue8m0_grid_binds_as_block_scales_of_its_tile() {
        for (shape, tile) in [(vec![2, 8], [32, 32]), (vec![64, 8], [1, 32]), (vec![1, 2], [64, 128])] {
            let directory = directory(Some(UE8M0), Some((Encoding::I8, shape)));
            let mut tensors = Tensors::new(&directory);
            let weight = tensors.bind_scaled(WEIGHT, &SHAPE).unwrap();
            assert_eq!(
                weight.transforms,
                [ImportTransform::ScaleBlocks { tensor: SCALE.into(), tile, codes: ScaleCodes::Ue8m0 }]
            );
            tensors.finish().unwrap();
        }
    }

    /// An undeclared or otherwise declared I8 grid, a grid that does not
    /// divide the matrix into power-of-two tiles, an F32 scale, and no scale
    /// at all are each refused, naming the tensor.
    #[test]
    fn any_other_scale_of_an_f8_e4m3_weight_is_refused_naming_it() {
        for (declared, scale) in [
            (None, Some((Encoding::I8, vec![2, 8]))),
            (Some("e8m0fnu"), Some((Encoding::I8, vec![2, 8]))),
            (Some(UE8M0), Some((Encoding::F32, vec![2, 8]))),
            (Some(UE8M0), Some((Encoding::F32, vec![1]))),
            (Some(UE8M0), Some((Encoding::I8, vec![3, 8]))),
            (Some(UE8M0), Some((Encoding::I8, vec![2, 8, 1]))),
            (Some(UE8M0), Some((Encoding::I8, vec![0, 8]))),
            (Some(UE8M0), None),
        ] {
            let directory = directory(declared, scale);
            let error = Tensors::new(&directory).bind_scaled(WEIGHT, &SHAPE).unwrap_err();
            assert!(error.to_string().contains(SCALE), "{error}");
        }
    }

    /// A grid whose inferred tile could be wrong is refused, naming it:
    /// `[300, 4096]` under `[10, 128]` (a partial last 32-row tile, inferred
    /// 30) and `[192, 256]` under `[2, 8]` (128-row tiles, inferred 96).
    #[test]
    fn a_grid_inferring_no_power_of_two_tile_is_refused_naming_it() {
        for (weight, grid) in [([300, 4096], vec![10, 128]), ([192, 256], vec![2, 8])] {
            let directory = shaped(weight, Some(UE8M0), Some((Encoding::I8, grid)));
            let error = Tensors::new(&directory).bind_scaled(WEIGHT, &weight).unwrap_err();
            assert!(error.to_string().contains(SCALE), "{error}");
        }
    }
}
