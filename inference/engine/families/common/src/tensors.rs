//! The strict tensor binder: every weight a definition names is bound by its
//! exact stored name and shape, and at the end every stored tensor must have
//! been bound.

use crate::{CompanionForm, HeaderError};
use magnitude_artifacts::gguf::{Directory, Encoding, TensorDescriptor};
use magnitude_family_contracts::{
    checked_product, ImportTransform, ScaleCodes, ScaledActivations, WeightDescriptor,
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
    /// weight's scale: of an F8_E4M3 matrix or stack, which takes one, its
    /// grid of block scales ([`Self::scale_grid`]); of any other weight,
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
        let (Some(stem), Some(name)) = (
            weight.strip_suffix(".weight"),
            ImportTransform::companion_scale(weight),
        ) else {
            return Ok(None);
        };
        let stored = self.stored(weight)?;
        if let Some(grid) = ImportTransform::scale_grid(weight, stored.encoding) {
            let scale = self.directory.tensor(&grid).ok_or(HeaderError::MissingWeight(grid))?;
            return self.scale_grid(stored, scale).map(Some);
        }
        let Some(scale) = self.directory.tensor(&name) else {
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

    /// Binds `scale` as the grid of block scales of the F8_E4M3 matrix, or
    /// stack of matrices, `weight` (`ImportTransform::ScaleBlocks`): `[rows
    /// / r, columns / c]` (after the stack's extent) for the tiles `[r, c]`
    /// it tiles each matrix by (`ImportTransform::grid_tile`), of the codes
    /// its stored type names (`ScaleCodes::of_stored`: I8 UE8M0 bytes, F32
    /// scales), so each grid of a file is read in its own codes. Any other
    /// scale is refused, naming it.
    fn scale_grid(
        &mut self,
        weight: &TensorDescriptor,
        scale: &'a TensorDescriptor,
    ) -> Result<ImportTransform, HeaderError> {
        let (stack, rows, columns) = match weight.shape[..] {
            [rows, columns] => (None, rows, columns),
            [stack, rows, columns] => (Some(stack), rows, columns),
            _ => {
                return Err(HeaderError::NotMatrix {
                    name: weight.name.clone(),
                    shape: weight.shape.clone(),
                });
            }
        };
        let (Some(codes), Some(tile)) = (
            ScaleCodes::of_stored(scale.encoding),
            ImportTransform::grid_tile(&weight.shape, &scale.shape),
        ) else {
            return Err(HeaderError::CompanionScale {
                name: scale.name.clone(),
                expected: CompanionForm::Grid { stack, rows, columns },
                encoding: scale.encoding,
                shape: scale.shape.clone(),
            });
        };
        self.bound.insert(scale.name.as_str());
        Ok(ImportTransform::ScaleBlocks {
            tensor: scale.name.clone(),
            tile,
            codes,
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
    use magnitude_artifacts::gguf::ByteOrder;

    const WEIGHT: &str = "blk.0.attn_q_b.weight";
    const SCALE: &str = "blk.0.attn_q_b.scale";

    const SHAPE: [u64; 2] = [64, 256];

    /// A file of an F8_E4M3 [`SHAPE`] weight and, when `scale` gives it,
    /// its `.scale`.
    fn directory(scale: Option<(Encoding, Vec<u64>)>) -> Directory {
        shaped(&SHAPE, scale)
    }

    fn shaped(weight: &[u64], scale: Option<(Encoding, Vec<u64>)>) -> Directory {
        stored(
            std::iter::once((WEIGHT, weight.to_vec(), Encoding::F8E4M3))
                .chain(scale.map(|(encoding, shape)| (SCALE, shape, encoding)))
                .collect(),
        )
    }

    /// A file of `tensors` (name, shape, encoding), with no metadata.
    fn stored(tensors: Vec<(&str, Vec<u64>, Encoding)>) -> Directory {
        Directory {
            version: 3,
            byte_order: ByteOrder::Little,
            alignment: 32,
            data_offset: 0,
            metadata: Vec::new(),
            tensors: tensors
                .into_iter()
                .map(|(name, shape, encoding)| TensorDescriptor {
                    name: name.into(),
                    shape,
                    encoding,
                    offset: 0,
                    nbytes: 0,
                })
                .collect(),
        }
    }

    /// A grid binds as block scales of its tile in the codes its stored
    /// type names: an I8 grid UE8M0 bytes, an F32 grid F32 scales.
    #[test]
    fn a_grid_binds_as_block_scales_of_its_tile_in_its_types_codes() {
        for codes in ScaleCodes::ALL {
            for (shape, tile) in [(vec![2, 8], [32, 32]), (vec![64, 8], [1, 32]), (vec![1, 2], [64, 128])] {
                let directory = directory(Some((codes.stored(), shape)));
                let mut tensors = Tensors::new(&directory);
                let weight = tensors.bind_scaled(WEIGHT, &SHAPE).unwrap();
                assert_eq!(weight.transforms, [ImportTransform::ScaleBlocks { tensor: SCALE.into(), tile, codes }]);
                tensors.finish().unwrap();
            }
        }
    }

    /// One file holds an I8-gridded and an F32-gridded weight, and each
    /// binds in its own grid's codes.
    #[test]
    fn one_file_binds_ue8m0_and_f32_grids_together() {
        let directory = stored(vec![
            (WEIGHT, SHAPE.to_vec(), Encoding::F8E4M3),
            (SCALE, vec![2, 8], Encoding::I8),
            ("blk.0.ffn_down.weight", vec![256, 512], Encoding::F8E4M3),
            ("blk.0.ffn_down.scale", vec![2, 4], Encoding::F32),
        ]);
        let mut tensors = Tensors::new(&directory);
        let ue8m0 = tensors.bind_scaled(WEIGHT, &SHAPE).unwrap();
        let f32 = tensors.bind_scaled("blk.0.ffn_down.weight", &[256, 512]).unwrap();
        tensors.finish().unwrap();
        assert_eq!(
            ue8m0.transforms,
            [ImportTransform::ScaleBlocks { tensor: SCALE.into(), tile: [32, 32], codes: ScaleCodes::Ue8m0 }]
        );
        assert_eq!(
            f32.transforms,
            [ImportTransform::ScaleBlocks {
                tensor: "blk.0.ffn_down.scale".into(),
                tile: [128, 128],
                codes: ScaleCodes::F32
            }]
        );
    }

    /// A stack of F8_E4M3 matrices binds with a grid of its matrices' grids,
    /// one tile for all of them; a grid over other matrices is refused.
    #[test]
    fn a_stacked_weight_binds_with_its_matrices_grids() {
        let stack = [3, 64, 256];
        let directory = shaped(&stack, Some((Encoding::F32, vec![3, 2, 8])));
        let mut tensors = Tensors::new(&directory);
        let weight = tensors.bind_scaled(WEIGHT, &stack).unwrap();
        assert_eq!(weight.shape, stack);
        assert_eq!(
            weight.transforms,
            [ImportTransform::ScaleBlocks { tensor: SCALE.into(), tile: [32, 32], codes: ScaleCodes::F32 }]
        );
        tensors.finish().unwrap();
        for grid in [vec![2, 2, 8], vec![2, 8], vec![3, 3, 8]] {
            let directory = shaped(&stack, Some((Encoding::F32, grid)));
            let error = Tensors::new(&directory).bind_scaled(WEIGHT, &stack).unwrap_err();
            assert!(error.to_string().contains(SCALE), "{error}");
        }
    }

    /// A grid of a type that holds no codes, a grid that does not divide
    /// the matrix into power-of-two tiles, and no scale at all are each
    /// refused, naming the tensor and the grid types that hold codes.
    #[test]
    fn any_other_scale_of_an_f8_e4m3_weight_is_refused_naming_it() {
        for scale in [
            Some((Encoding::F16, vec![2, 8])),
            Some((Encoding::BF16, vec![2, 8])),
            Some((Encoding::F32, vec![1])),
            Some((Encoding::I8, vec![3, 8])),
            Some((Encoding::I8, vec![2, 8, 1])),
            Some((Encoding::I8, vec![0, 8])),
            Some((Encoding::F32, vec![3, 8])),
            None,
        ] {
            let error = Tensors::new(&directory(scale.clone())).bind_scaled(WEIGHT, &SHAPE).unwrap_err();
            assert!(error.to_string().contains(SCALE), "{error}");
            if scale.is_some() {
                assert!(error.to_string().contains("an I8 or F32 grid"), "{error}");
            }
        }
    }

    /// A grid whose inferred tile could be wrong is refused, naming it,
    /// under either codes:
    /// `[300, 4096]` under `[10, 128]` (a partial last 32-row tile, inferred
    /// 30) and `[192, 256]` under `[2, 8]` (128-row tiles, inferred 96).
    #[test]
    fn a_grid_inferring_no_power_of_two_tile_is_refused_naming_it() {
        for codes in ScaleCodes::ALL {
            for (weight, grid) in [([300, 4096], vec![10, 128]), ([192, 256], vec![2, 8])] {
                let directory = shaped(&weight, Some((codes.stored(), grid)));
                let error = Tensors::new(&directory).bind_scaled(WEIGHT, &weight).unwrap_err();
                assert!(error.to_string().contains(SCALE), "{error}");
            }
        }
    }
}
