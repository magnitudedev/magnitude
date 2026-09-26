use super::super::*;
use super::*;

impl<'a> QualificationView<'a> {
    pub(super) fn qualify_shape_rows(&self, device: &Device) -> Result<(), CatalogFailure> {
        // One vocabulary row at unit temperature without cuts or penalties:
        // shaping is the identity.
        let vocabulary = self.geometry.vocabulary;
        let values = (0..vocabulary)
            .map(|token| (token % 4) as f32 - 1.0)
            .collect::<Vec<_>>();
        let params = [1.0_f32, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let history = [-1_i32; 64];
        let logits = tensor_f32(device, &[1, vocabulary], &values, "shape_rows", "fixed")?;
        let params = tensor_f32(device, &[1, 8], &params, "shape_rows", "fixed")?;
        let history_bytes = history
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let history = Tensor::from_host(device, Element::i32(), &[1, 64], &history_bytes)
            .map_err(|error| qualification("shape_rows", "fixed", error))?;
        let mut out = Tensor::zeros(device, Element::f32(), &[1, vocabulary])
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
        let expected = values
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

    pub(super) fn qualify_sample_rows(&self, device: &Device) -> Result<(), CatalogFailure> {
        // Logits ascend with the token. Row 0 is unconstrained (its empty
        // mask is ignored) and selects the last token; row 1 admits only
        // token 0.
        let vocabulary = self.geometry.vocabulary;
        let words = vocabulary.div_ceil(32);
        let values = (0..2 * vocabulary)
            .map(|index| (index % vocabulary) as f32)
            .collect::<Vec<_>>();
        let logits = tensor_f32(device, &[2, vocabulary], &values, "sample_rows", "fixed")?;
        let mask_words = (0..2 * words)
            .map(|index| u32::from(index == words))
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        let mask = Tensor::from_host(device, Element::u32(), &[2, words], &mask_words)
            .map_err(|error| qualification("sample_rows", "fixed", error))?;
        let flags = [0_i32, 1_i32]
            .iter()
            .flat_map(|flag| flag.to_le_bytes())
            .collect::<Vec<_>>();
        let constrained = Tensor::from_host(device, Element::i32(), &[2], &flags)
            .map_err(|error| qualification("sample_rows", "fixed", error))?;
        let draws = Tensor::zeros(device, Element::u32(), &[2, 6])
            .map_err(|error| qualification("sample_rows", "fixed", error))?;
        let mut result = Tensor::zeros(device, Element::i32(), &[2, 2])
            .map_err(|error| qualification("sample_rows", "fixed", error))?;
        self.programs
            .target
            .sample
            .call(sample_rows::Args {
                logits: &logits,
                mask: &mask,
                constrained: &constrained,
                draws: &draws,
                result: &mut result,
            })
            .map_err(|error| qualification("sample_rows", "fixed", error))?;
        let last = i32::try_from(vocabulary - 1)
            .map_err(|error| qualification("sample_rows", "fixed", error))?;
        let expected = [last, 0, 0, 0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        if result
            .read_to_host()
            .map_err(|error| qualification("sample_rows", "fixed", error))?
            != expected
        {
            return Err(qualification(
                "sample_rows",
                "fixed",
                "greedy smoke mismatch",
            ));
        }
        Ok(())
    }

    pub(super) fn qualify_conditioning_overlay(
        &self,
        device: &Device,
    ) -> Result<(), CatalogFailure> {
        let input = tensor_f32(
            device,
            &[1, 2],
            &[1.0, -2.0],
            "conditioning_overlay",
            "fixed",
        )?;
        let mut out = Tensor::zeros(device, Element::f32(), &[1, 2])
            .map_err(|error| qualification("conditioning_overlay", "fixed", error))?;
        self.programs
            .state
            .conditioning
            .as_ref()
            .expect("attested conditioning slot")
            .call(conditioning_overlay::Args {
                input: &input,
                out: &mut out,
            })
            .map_err(|error| qualification("conditioning_overlay", "fixed", error))?;
        if out
            .read_to_host()
            .map_err(|error| qualification("conditioning_overlay", "fixed", error))?
            != [1.0_f32.to_le_bytes(), (-2.0_f32).to_le_bytes()].concat()
        {
            return Err(qualification(
                "conditioning_overlay",
                "fixed",
                "copy smoke mismatch",
            ));
        }
        Ok(())
    }

    pub(super) fn qualify_copy_rows(
        &self,
        device: &Device,
        element: Element,
        binding: &'static str,
    ) -> Result<(), CatalogFailure> {
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
