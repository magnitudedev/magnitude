mod glue;
mod head;
mod helpers;
mod target;
mod vision;

use helpers::*;

use super::*;

/// Runs every attested entry once on fixtures at the model's geometry (the
/// entries are specialized to it) with weights chosen so the result is known.
pub(super) struct QualificationView<'a> {
    programs: &'a AttestedPrograms,
    plan: &'a ProgramPlan,
    geometry: &'a magnitude_model_contracts::DecoderGeometry,
    load: &'a crate::ModelLoadPlan,
}

impl<'a> QualificationView<'a> {
    pub(super) fn new(
        programs: &'a AttestedPrograms,
        plan: &'a ProgramPlan,
        geometry: &'a magnitude_model_contracts::DecoderGeometry,
        load: &'a crate::ModelLoadPlan,
    ) -> Self {
        Self {
            programs,
            plan,
            geometry,
            load,
        }
    }

    /// The planned shape of one weight, for fixtures of its geometry.
    fn weight_shape(
        &self,
        scope: magnitude_model_contracts::WeightScope,
        kind: magnitude_model_contracts::WeightKind,
        entry: &'static str,
    ) -> Result<&'a [u64], CatalogFailure> {
        self.load
            .weights()
            .find(|weight| weight.role.scope == scope && weight.role.kind == kind)
            .map(|weight| weight.shape.as_slice())
            .ok_or_else(|| {
                qualification(entry, "fixture", format!("the load plan has no {kind:?} weight in {scope:?}"))
            })
    }

    pub(super) fn qualify(&self, device: &Device) -> Result<(), CatalogFailure> {
        for (slot, handle) in &self.programs.imports {
            let (
                ImportProgramSlot::Dense {
                    source: source_dtype,
                    resident: target_dtype,
                },
                AttestedImport::Dense(handle),
            ) = (slot, handle)
            else {
                continue;
            };
            let bindings = dense_binding_name(*source_dtype, *target_dtype);
            let source_bytes = one_bytes(*source_dtype);
            let source =
                Tensor::from_host(device, Element::dense(*source_dtype), &[1, 1, 1], &source_bytes)
                    .map_err(|error| qualification_dynamic("import_dense", &bindings, error))?;
            let destination = handle
                .call(import_dense::Args { source: &source })
                .map(|results| results.value)
                .map_err(|error| qualification_dynamic("import_dense", &bindings, error))?;
            let output = destination
                .read_to_host()
                .map_err(|error| qualification_dynamic("import_dense", &bindings, error))?;
            if destination.element() != Element::dense(*target_dtype)
                || output != one_bytes(*target_dtype)
            {
                return Err(qualification_dynamic(
                    "import_dense",
                    &bindings,
                    "incorrect identity conversion",
                ));
            }
        }
        for (slot, handle) in &self.programs.imports {
            let (
                ImportProgramSlot::Repack {
                    source: source_element,
                    resident: target_element,
                },
                AttestedImport::Repack(handle),
            ) = (slot, handle)
            else {
                continue;
            };
            // Two packets per row of 17 rows (a partial 16-row tile), of
            // arbitrary source bytes, against the registry's host reference.
            let bindings = element_binding_name(*source_element, *target_element);
            let failure = |error: String| qualification_dynamic("repack_weight", &bindings, error);
            let group = source_element
                .logical_group()
                .ok_or_else(|| failure("the repack source is not packet storage".into()))?;
            let shape = [1, 17, 2 * group];
            let length = source_element
                .canonical_byte_len(&shape)
                .map_err(|error| failure(error.to_string()))?;
            let bytes = (0..length)
                .map(|index| (index as u8).wrapping_mul(37).wrapping_add(11))
                .collect::<Vec<_>>();
            let source = Tensor::from_host(device, *source_element, &shape, &bytes)
                .map_err(|error| failure(error.to_string()))?;
            let result = handle
                .call(repack_weight::Args { source: &source })
                .map(|results| results.value)
                .map_err(|error| failure(error.to_string()))?;
            let output = result
                .read_to_host()
                .map_err(|error| failure(error.to_string()))?;
            let expected = target_element
                .repack_host(*source_element, &shape, &bytes)
                .ok_or_else(|| failure("no registered conversion for the binding".into()))?;
            if result.element() != *target_element || output != expected {
                return Err(failure("repack differs from the registered conversion".into()));
            }
        }
        self.qualify_shape_rows(device)?;
        self.qualify_sample_rows(device)?;
        self.qualify_conditioning_overlay(device)?;
        self.qualify_target(device)?;
        self.qualify_head(device)?;
        self.qualify_vision(device)?;
        for (element, binding) in [
            (Element::f32(), "A=f32"),
            (Element::f16(), "A=f16"),
            (Element::bf16(), "A=bf16"),
            (Element::u32(), "A=u32"),
        ] {
            if self.plan.state().copies().contains(&element) {
                self.qualify_copy_rows(device, element, binding)?;
            }
        }
        Ok(())
    }
}
