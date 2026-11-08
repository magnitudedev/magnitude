//! A vocabulary projection's stored form.
//!
//! A progressive head (`ImportTransform::Progressive`) is stored as its
//! planes wherever the backend has the kernels that read them
//! (`reads_progressive_heads`): a selection of up to the backend's
//! `certified_rows` runs the certified levels (`readout_top_rows`,
//! `readout_refine_rows`, `readout_exact_rows`), which read the vocabulary
//! rows the earlier bounds keep, and every other read runs the planes' full
//! pass (`readout_planes_rows`). Packed is the form of every other head, and
//! of a scope whose statics lie outside the levels' domain. The form is
//! structural: it is never measured or searched.
//!
//! The measured basis for the rule. The planes' full pass against packed,
//! one row, RTX 3090 CUDA, Qwen3.6-35B-A3B q4: 0.70 against 2.02 ms a decode
//! step. Readout device time per real decode step, certified levels over
//! packed, on two models and two fixtures (an M4 Pro, 4,096 context):
//!   Qwen3.6-35B-A3B  prose-repeat 0.64, prose-continue 0.80
//!     (run 20261010T123257Z-a3b.diag-31f355)
//!   Nemotron 3.5 Lightning 30B-A3B Q4  prose-repeat 0.67, prose-continue 0.89
//!     (run 20261010T124501Z-a3b.diag-96f42c)

use super::specialization::{entry_statics, prepare_program};
use super::{CatalogFailure, ProgressiveReadoutKernels};
use magnitude_family_contracts::ProgressiveCodeDepth;
use magnitude_kernels::{
    readout_exact_rows, readout_head_rows, readout_planes_rows, readout_project_rows,
    readout_refine_rows, readout_top_rows,
};
use seismic::{BackendName, BoundEntry, Element, Entry, NativeKernel, NativeSpecialization};
use std::{cell::RefCell, rc::Rc};

/// The chosen implementation of a scope-owned vocabulary projection.
#[derive(Clone, Debug)]
pub enum ProjectionKernels {
    RawPacked(NativeKernel<readout_head_rows::Entry>),
    PreparedPacked(NativeKernel<readout_project_rows::Entry>),
    /// A scope stored as its planes, and its reader's packed kernels.
    Qualified {
        packed: Box<ProjectionKernels>,
        progressive: ProgressiveReadoutKernels,
    },
}

/// One reader's progressive recipe: the statics its levels are formed at.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ReadoutRecipe {
    pub norm: Element,
    pub activation: Element,
    pub vocabulary: u64,
    pub hidden: u64,
    /// The readout's input form (`PT`, `NM`): partial rows summed, and the
    /// norm's extent (0 for the hyper-connection exit site's partials).
    pub parts: u64,
    pub norm_extent: u64,
    /// The progressive head's code depth (`ImportTransform::Progressive`).
    pub depth: ProgressiveCodeDepth,
    pub scale_extent: u64,
}

/// A recipe's formed levels, shared by every reader whose recipe forms the
/// same programs.
pub(crate) type RecipePreparation = Rc<RefCell<Option<ProgressiveReadoutKernels>>>;

fn fail(outcome: impl ToString) -> CatalogFailure {
    CatalogFailure::Preparation {
        entry: "readout_recipe",
        bindings: "progressive".into(),
        outcome: outcome.to_string(),
    }
}

impl ReadoutRecipe {
    /// The recipe's value of every static its kernels declare; each takes
    /// only the statics it declares.
    fn values(self) -> [(&'static str, u64); 8] {
        [
            ("V", self.vocabulary),
            ("D", self.hidden),
            ("PT", self.parts),
            ("NM", self.norm_extent),
            ("P", u64::from(self.depth.rest_planes())),
            ("WS", self.scale_extent),
            ("S", self.vocabulary),
            ("I", 0),
        ]
    }

    /// `entry`'s statics in this recipe on `backend`.
    fn statics<E: Entry>(
        self,
        backend: BackendName,
        entry: &BoundEntry<E>,
    ) -> Result<NativeSpecialization, CatalogFailure> {
        entry_statics(backend, entry, "readout_recipe", &self.values())
    }

    fn top_entry(self) -> BoundEntry<readout_top_rows::Entry> {
        readout_top_rows::native_entry_with(readout_top_rows::Elements {
            NW: self.norm,
            A: self.activation,
        })
    }

    fn refine_entry(self) -> BoundEntry<readout_refine_rows::Entry> {
        readout_refine_rows::native_entry_with(readout_refine_rows::Elements { A: self.activation })
    }

    fn exact_entry(self) -> BoundEntry<readout_exact_rows::Entry> {
        readout_exact_rows::native_entry_with(readout_exact_rows::Elements { A: self.activation })
    }

    fn planes_entry(self) -> BoundEntry<readout_planes_rows::Entry> {
        readout_planes_rows::native_entry_with(readout_planes_rows::Elements {
            NW: self.norm,
            A: self.activation,
        })
    }

    /// Whether every level admits the recipe's statics on `backend`. A
    /// static outside a level's domain leaves the scope packed; any other
    /// failure is a fault.
    pub(crate) fn admitted(self, backend: BackendName) -> Result<bool, CatalogFailure> {
        let admitted = (|| {
            self.statics(backend, &self.top_entry())?;
            self.statics(backend, &self.refine_entry())?;
            self.statics(backend, &self.exact_entry())?;
            self.statics(backend, &self.planes_entry())?;
            Ok(())
        })();
        match admitted {
            Ok(()) => Ok(true),
            Err(CatalogFailure::KernelDomain { .. }) => Ok(false),
            Err(error) => Err(error),
        }
    }

    /// The identity of `entry`'s program in this recipe on `device`.
    fn digest<E: Entry>(
        self,
        device: &seismic::Device,
        entry: &BoundEntry<E>,
    ) -> Result<String, CatalogFailure> {
        entry
            .digest(device, &self.statics(device.backend(), entry)?)
            .map_err(fail)
    }

    /// The identity of the recipe's levels on `device`: readers whose
    /// recipes digest alike share one formation.
    pub(crate) fn digests(self, device: &seismic::Device) -> Result<[String; 4], CatalogFailure> {
        Ok([
            self.digest(device, &self.top_entry())?,
            self.digest(device, &self.refine_entry())?,
            self.digest(device, &self.exact_entry())?,
            self.digest(device, &self.planes_entry())?,
        ])
    }

    /// The recipe's levels, formed once per `preparation`.
    pub(crate) fn form_shared(
        self,
        device: &seismic::Device,
        preparation: &RecipePreparation,
    ) -> Result<ProgressiveReadoutKernels, CatalogFailure> {
        if let Some(progressive) = preparation.borrow().clone() {
            return Ok(progressive);
        }
        let formed = self.form(device)?;
        *preparation.borrow_mut() = Some(formed.clone());
        Ok(formed)
    }

    /// `entry` prepared at its declared defaults at this recipe's statics.
    fn at_defaults<E: Entry>(
        self,
        device: &seismic::Device,
        entry: BoundEntry<E>,
    ) -> Result<NativeKernel<E>, CatalogFailure> {
        let statics = self.statics(device.backend(), &entry)?;
        let defaults = entry
            .default_specialization(device, &statics)
            .map_err(fail)?
            .ok_or_else(|| fail("readout recipe kernel admits no default at its statics"))?;
        prepare_program(device, entry, &defaults).map_err(fail)
    }

    fn form(self, device: &seismic::Device) -> Result<ProgressiveReadoutKernels, CatalogFailure> {
        Ok(ProgressiveReadoutKernels {
            top: self.at_defaults(device, self.top_entry())?,
            refine: self.at_defaults(device, self.refine_entry())?,
            exact: self.at_defaults(device, self.exact_entry())?,
            planes: self.at_defaults(device, self.planes_entry())?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recipe(vocabulary: u64, parts: u64, norm_extent: u64) -> ReadoutRecipe {
        ReadoutRecipe {
            norm: Element::bf16(),
            activation: Element::bf16(),
            vocabulary,
            hidden: 2048,
            parts,
            norm_extent,
            depth: ProgressiveCodeDepth::Eight,
            scale_extent: 0,
        }
    }

    /// Every level of the recipe digests and forms on each backend with
    /// its levels, in both input forms: a normed exit's one row, and a
    /// hyper-connection exit's partial rows without a norm.
    #[test]
    fn every_recipe_level_forms_at_its_own_statics() {
        let catalog = seismic::Devices::discover(&[]).unwrap();
        let available = [BackendName::Metal, BackendName::Cuda]
            .into_iter()
            .filter(|&backend| crate::test_support::available(&catalog, backend));
        for backend in available {
            let device = crate::test_support::open_backend(&catalog, backend)
                .unwrap_or_else(|error| panic!("an available {backend:?} executor opens: {error}"));
            for (parts, norm_extent) in [(1, 2048), (4, 0)] {
                let recipe = recipe(4096, parts, norm_extent);
                let form = format!("{backend:?} PT={parts} NM={norm_extent}");
                recipe
                    .digests(&device)
                    .unwrap_or_else(|error| panic!("{form}: {error:?}"));
                recipe
                    .form(&device)
                    .unwrap_or_else(|error| panic!("{form}: {error:?}"));
            }
        }
    }

    /// Every backend with the levels admits the recipe at the heads of
    /// Qwen3.5-2B (248,320 x 2,048) and LFM3-1.7B (128,000 x 2,048), each
    /// tied to its Q8_0 token embedding, without a device.
    #[test]
    fn every_backend_with_the_levels_admits_the_recipe() {
        for backend in BackendName::ALL {
            if !crate::programs::graph::readout::reads_progressive_heads(backend).unwrap() {
                continue;
            }
            for vocabulary in [248_320, 128_000] {
                assert!(
                    recipe(vocabulary, 1, 2048).admitted(backend).unwrap(),
                    "{backend:?} V={vocabulary}"
                );
            }
        }
    }
}
