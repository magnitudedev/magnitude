use super::super::*;
use super::*;

impl<'a> QualificationView<'a> {
    pub(super) fn qualify_shape_rows(&self, device: &Device) -> Result<(), CatalogError> {
        let logits = [-1.0_f32, 0.0, 1.0, 2.0];
        let params = [1.0_f32, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let history = [-1_i32; 64];
        let logits = tensor_f32(device, &[1, 4], &logits, "shape_rows", "fixed")?;
        let params = tensor_f32(device, &[1, 8], &params, "shape_rows", "fixed")?;
        let history_bytes = history
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let history = Tensor::from_host(device, Element::i32(), &[1, 64], &history_bytes)
            .map_err(|error| qualification("shape_rows", "fixed", error))?;
        let mut out = Tensor::zeros(device, Element::f32(), &[1, 4])
            .map_err(|error| qualification("shape_rows", "fixed", error))?;
        self.programs
            .target
            .shape
            .call(shape_rows::Args {
                logits: &logits,
                params: &params,
                history: &history,
                out: &mut out,
            })
            .map_err(|error| qualification("shape_rows", "fixed", error))?;
        let output = out
            .read_to_host()
            .map_err(|error| qualification("shape_rows", "fixed", error))?;
        let expected = [-1.0_f32, 0.0, 1.0, 2.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        if output != expected {
            return Err(qualification(
                "shape_rows",
                "fixed",
                "identity smoke mismatch",
            ));
        }
        Ok(())
    }

    pub(super) fn qualify_sample_rows(&self, device: &Device) -> Result<(), CatalogError> {
        let logits = tensor_f32(device, &[1, 2], &[1.0, 2.0], "sample_rows", "fixed")?;
        let mask = Tensor::from_host(device, Element::u32(), &[1, 1], &3_u32.to_le_bytes())
            .map_err(|error| qualification("sample_rows", "fixed", error))?;
        let draws = Tensor::zeros(device, Element::u32(), &[1, 6])
            .map_err(|error| qualification("sample_rows", "fixed", error))?;
        let mut result = Tensor::zeros(device, Element::i32(), &[1, 2])
            .map_err(|error| qualification("sample_rows", "fixed", error))?;
        self.programs
            .target
            .sample
            .call(sample_rows::Args {
                logits: &logits,
                mask: &mask,
                draws: &draws,
                result: &mut result,
            })
            .map_err(|error| qualification("sample_rows", "fixed", error))?;
        if result
            .read_to_host()
            .map_err(|error| qualification("sample_rows", "fixed", error))?
            != [1_i32.to_le_bytes(), 0_i32.to_le_bytes()].concat()
        {
            return Err(qualification(
                "sample_rows",
                "fixed",
                "greedy smoke mismatch",
            ));
        }
        Ok(())
    }

    pub(super) fn qualify_conditioning_overlay(&self, device: &Device) -> Result<(), CatalogError> {
        let input = tensor_f32(
            device,
            &[1, 2],
            &[1.0, -2.0],
            "qwen_conditioning_overlay",
            "fixed",
        )?;
        let mut out = Tensor::zeros(device, Element::f32(), &[1, 2])
            .map_err(|error| qualification("qwen_conditioning_overlay", "fixed", error))?;
        self.programs
            .state
            .conditioning
            .as_ref()
            .expect("attested conditioning slot")
            .call(qwen_conditioning_overlay::Args {
                input: &input,
                out: &mut out,
            })
            .map_err(|error| qualification("qwen_conditioning_overlay", "fixed", error))?;
        if out
            .read_to_host()
            .map_err(|error| qualification("qwen_conditioning_overlay", "fixed", error))?
            != [1.0_f32.to_le_bytes(), (-2.0_f32).to_le_bytes()].concat()
        {
            return Err(qualification(
                "qwen_conditioning_overlay",
                "fixed",
                "copy smoke mismatch",
            ));
        }
        Ok(())
    }

    pub(super) fn qualify_gather_rows(&self, device: &Device) -> Result<(), CatalogError> {
        let source = tensor_f32(
            device,
            &[3, 2],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "gather_rows",
            "fixed",
        )?;
        let rows = Tensor::from_host(
            device,
            Element::i32(),
            &[2],
            &[2_i32.to_le_bytes(), 0_i32.to_le_bytes()].concat(),
        )
        .map_err(|error| qualification("gather_rows", "fixed", error))?;
        let output = self
            .programs
            .state
            .gather
            .as_ref()
            .expect("attested gather slot")
            .call(gather_rows::Args {
                source: &source,
                rows: &rows,
            })
            .map_err(|error| qualification("gather_rows", "fixed", error))?
            .value
            .read_to_host()
            .map_err(|error| qualification("gather_rows", "fixed", error))?;
        let expected = [5.0_f32, 6.0, 1.0, 2.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        if output != expected {
            return Err(qualification("gather_rows", "fixed", "row order mismatch"));
        }
        Ok(())
    }

    pub(super) fn qualify_copy_rows(
        &self,
        device: &Device,
        element: Element,
        binding: &'static str,
    ) -> Result<(), CatalogError> {
        let width = element.dtype().expect("dense qualified binding").bytes() as usize;
        let source = (0..4 * width).map(|byte| byte as u8).collect::<Vec<_>>();
        let sentinel = vec![0xff; 4 * width];
        let src = Tensor::from_host(device, element, &[2, 1, 2], &source)
            .map_err(|error| qualification("copy_rows", binding, error))?;
        let mut dst = Tensor::from_host(device, element, &[2, 1, 2], &sentinel)
            .map_err(|error| qualification("copy_rows", binding, error))?;
        let from = Tensor::from_host(device, Element::i32(), &[1], &1_i32.to_le_bytes())
            .map_err(|error| qualification("copy_rows", binding, error))?;
        let to = Tensor::from_host(device, Element::i32(), &[1], &0_i32.to_le_bytes())
            .map_err(|error| qualification("copy_rows", binding, error))?;
        let (_, copy) = self
            .programs
            .state
            .copies
            .iter()
            .find(|(available, _)| *available == element)
            .expect("attested copy slot");
        copy.call(copy_rows::Args {
            src: &src,
            dst: &mut dst,
            from: &from,
            to: &to,
        })
        .map_err(|error| qualification("copy_rows", binding, error))?;
        let output = dst
            .read_to_host()
            .map_err(|error| qualification("copy_rows", binding, error))?;
        if output[..2 * width] != source[2 * width..]
            || output[2 * width..] != sentinel[2 * width..]
        {
            return Err(qualification(
                "copy_rows",
                binding,
                "indexed row smoke mismatch",
            ));
        }
        Ok(())
    }
}
