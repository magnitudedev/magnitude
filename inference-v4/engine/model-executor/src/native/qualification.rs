mod glue;
mod head;
mod helpers;
mod target;
mod vision;

use helpers::*;

use super::*;

pub(super) struct QualificationView<'a> {
    programs: &'a AttestedPrograms,
    plan: &'a ProgramPlan,
}

impl<'a> QualificationView<'a> {
    pub(super) fn new(programs: &'a AttestedPrograms, plan: &'a ProgramPlan) -> Self {
        Self { programs, plan }
    }

    pub(super) fn qualify(&self, device: &Device) -> Result<(), CatalogError> {
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
                Tensor::from_host(device, Element::dense(*source_dtype), &[1], &source_bytes)
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
            let (_, _, bindings, logical, source_bytes) = repack_bindings()
                .into_iter()
                .find(|(source, resident, _, _, _)| {
                    *source == *source_element && *resident == *target_element
                })
                .ok_or_else(|| {
                    qualification_dynamic(
                        "repack_weight",
                        &element_binding_name(*source_element, *target_element),
                        "no bounded qualification fixture exists for the required binding",
                    )
                })?;
            let source =
                Tensor::from_host(device, *source_element, &[logical], &vec![0; source_bytes])
                    .map_err(|error| qualification("repack_weight", bindings, error))?;
            let result = handle
                .call(repack_weight::Args { source: &source })
                .map(|results| results.value)
                .map_err(|error| qualification("repack_weight", bindings, error))?;
            let output = result
                .read_to_host()
                .map_err(|error| qualification("repack_weight", bindings, error))?;
            // An all-zero IQ4_XS packet has a zero base scale and signed
            // sub-scales of -32. Its exact resident factor is therefore -0.0,
            // not the all-zero byte pattern used by the other formats.
            let expected = if source_element.name() == "gguf_iq4_xs" {
                let mut bytes = vec![0; 160];
                for group in 0..8 {
                    bytes[128 + group * 4..132 + group * 4]
                        .copy_from_slice(&(-0.0_f32).to_le_bytes());
                }
                bytes
            } else {
                vec![0; output.len()]
            };
            if result.element() != *target_element || output != expected {
                return Err(qualification(
                    "repack_weight",
                    bindings,
                    "zero-packet repack mismatch",
                ));
            }
        }
        self.qualify_shape_rows(device)?;
        self.qualify_sample_rows(device)?;
        self.qualify_conditioning_overlay(device)?;
        self.qualify_gather_rows(device)?;
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
