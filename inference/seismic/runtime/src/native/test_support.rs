//! Fixtures the tuner's tests and the survey replay's tests share: native
//! declarations checked for Metal, V4.1's Metal attention_decode as
//! declared, and a planned domain whose placements are counted.

use super::plan::{self, ParameterAddress, PlanError};
use super::search::LaunchKey;
use seismic_lang::checked::{NativeImplementation, NativeSpecialization, SourceFile, SourceSet};
use seismic_lang::registry::BackendName;
use std::cell::Cell;

/// The Metal implementation of a one-entry module whose native declaration
/// body is `body`, checked with a source whose kernels take no template
/// parameters.
pub(crate) fn declared(body: &str) -> NativeImplementation {
    let module = seismic_lang::native_source::check_fixture(
        SourceSet::new(vec![SourceFile {
            path: "scale.seismic".into(),
            text: format!(
                "fn scale[N](x: &tensor[N] f32) -> tensor[N] f32:\n    return to_owned(x)\n\n\
                 native scale for metal from \"scale.metal\":\n{body}"
            ),
        }]),
        &[],
    )
    .unwrap();
    module
        .native_implementation(module.entries()[0].id, BackendName::Metal)
        .unwrap()
        .clone()
}

/// V4.1's Metal attention_decode parameters as declared, restricted as
/// its `where` restricts them at V4.1's shapes: 129,024 declared
/// assignments, 28 admissible (SPAN and PARTS free), the restrictions
/// coupling the last parameters to the form.
pub(crate) const V41_ATTENTION_DECODE: &str = concat!(
    "    params (SPAN in [32, 64, 128, 256] default 32, PARTS in [16, 8, 32, 64, 128, 256, 512] default 16, \
     SIMDS in [4, 8, 2, 16] default 4, SLICES in [1, 2, 4, 8] default 1, form MATRIX in [0, 1, 2] default 2, \
     KEYS in [16, 8, 32] default 8, TOKENS in [1, 2, 4, 8] default 1, \
     HEADS in [1, 2, 4, 8, 16, 32, 64, 128] default 64)\n",
    "    where SIMDS % SLICES == 0 and (MATRIX >= 1 or (KEYS == 16 and TOKENS == 1)) \
     and (MATRIX < 2 or (TOKENS == 1 and SIMDS == 4 and HEADS >= 64 and HEADS / 2 < 64)) \
     and (MATRIX == 0 or SLICES == 1) and MATRIX == 2 and KEYS == 8\n",
    "    launch decode:\n",
    "        threadgroups (ceil_div(N, SPAN * SLICES * KEYS * TOKENS), PARTS * HEADS, MATRIX + 1)\n",
    "        threads_per_threadgroup (SIMDS * 32, 1, 1)\n",
);

/// A planned domain whose placements are counted.
pub(crate) struct CountedDomain<D> {
    pub domain: D,
    pub placed: Cell<usize>,
}

impl<D> CountedDomain<D> {
    pub fn new(domain: D) -> Self {
        Self { domain, placed: Cell::new(0) }
    }
}

impl<D: plan::Domain> plan::Domain for CountedDomain<D> {
    fn place(
        &self,
        base: &NativeSpecialization,
        parameters: &[ParameterAddress],
        values: &[u64],
    ) -> Result<Option<NativeSpecialization>, PlanError> {
        self.placed.set(self.placed.get() + 1);
        self.domain.place(base, parameters, values)
    }

    fn nearest(
        &self,
        base: &NativeSpecialization,
        parameters: &[ParameterAddress],
        values: &[u64],
    ) -> Result<Option<NativeSpecialization>, PlanError> {
        self.domain.nearest(base, parameters, values)
    }

    fn admissible(
        &self,
        base: &NativeSpecialization,
        parameters: &[ParameterAddress],
        after: Option<&[u64]>,
        limit: usize,
    ) -> Result<Vec<Vec<u64>>, PlanError> {
        self.domain.admissible(base, parameters, after, limit)
    }

    fn forecast(
        &self,
        point: usize,
        baseline: &LaunchKey,
        specialization: &NativeSpecialization,
    ) -> LaunchKey {
        self.domain.forecast(point, baseline, specialization)
    }
}
