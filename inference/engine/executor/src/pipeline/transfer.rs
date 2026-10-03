//! Bounded PCIe activation storage. This owns no model weights or state.
use super::PipelineRefusal;
use seismic::{Device, Element, Tensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ActivationContract {
    max_rows: u64,
    hidden: u64,
}
impl ActivationContract {
    pub fn new(max_rows: u64, hidden: u64) -> Result<Self, PipelineRefusal> {
        if !matches!(max_rows, 1 | 2) || hidden == 0 {
            return Err(PipelineRefusal::UnsupportedProfile);
        }
        let contract = Self { max_rows, hidden };
        contract.bytes(max_rows)?;
        Ok(contract)
    }
    pub fn bytes(self, rows: u64) -> Result<u64, PipelineRefusal> {
        if !matches!(rows, 1 | 2) || rows > self.max_rows {
            return Err(PipelineRefusal::UnsupportedProfile);
        }
        rows.checked_mul(self.hidden)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| PipelineRefusal::Preparation("activation bytes overflow".into()))
    }
    pub fn validate(
        self,
        rows: u64,
        element: Element,
        bytes: usize,
    ) -> Result<(), PipelineRefusal> {
        if element != Element::f32() || u64::try_from(bytes).ok() != Some(self.bytes(rows)?) {
            return Err(PipelineRefusal::Preparation(
                "activation representation or byte count differs".into(),
            ));
        }
        Ok(())
    }
}

pub(crate) struct ActivationBuffer {
    contract: ActivationContract,
    tensor: Tensor,
}
impl ActivationBuffer {
    /// Caller holds the exact local device and host staging claims before this
    /// allocation. Reuse avoids allocating device scratch during generation.
    pub fn allocate(
        device: &Device,
        contract: ActivationContract,
    ) -> Result<Self, PipelineRefusal> {
        let tensor = Tensor::zeros(
            device,
            Element::f32(),
            &[contract.max_rows, contract.hidden],
        )
        .map_err(|e| PipelineRefusal::Preparation(e.to_string()))?;
        Ok(Self { contract, tensor })
    }
    pub(crate) fn storage_bytes(&self) -> u64 {
        self.tensor.storage_bytes()
    }
    pub fn receive(
        &mut self,
        device: &Device,
        rows: u64,
        completed: &[u8],
    ) -> Result<Tensor, PipelineRefusal> {
        self.contract
            .validate(rows, Element::f32(), completed.len())?;
        if !self.tensor.device().same_device(device) {
            return Err(PipelineRefusal::ForeignDevice);
        }
        let mut view = self
            .tensor
            .slice_leading(0, rows)
            .map_err(|e| PipelineRefusal::Preparation(e.to_string()))?;
        view.write_from_host(completed)
            .map_err(|e| PipelineRefusal::Preparation(e.to_string()))?;
        Ok(view)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn receive_reuses_bounded_storage_and_refuses_foreign_owner() {
        let catalog = seismic::DeviceCatalog::discover().unwrap();
        let device = catalog.open_backend(seismic::BackendName::Cpu).unwrap();
        let foreign = seismic::DeviceCatalog::discover()
            .unwrap()
            .open_backend(seismic::BackendName::Cpu)
            .unwrap();
        let mut buffer =
            ActivationBuffer::allocate(&device, ActivationContract::new(2, 4).unwrap()).unwrap();
        let before = buffer.storage_bytes();
        assert_eq!(before, 32);
        for rows in [2, 1, 2, 1] {
            let bytes = (0..rows * 4)
                .flat_map(|i| (i as f32 + rows as f32).to_le_bytes())
                .collect::<Vec<_>>();
            let received = buffer.receive(&device, rows, &bytes).unwrap();
            assert_eq!(received.extents(), &[rows, 4]);
            assert!(received.device().same_device(&device));
            assert_eq!(received.read_to_host().unwrap(), bytes);
            assert_eq!(buffer.tensor.storage_bytes(), before);
            assert!(matches!(
                buffer.receive(&foreign, rows, &bytes),
                Err(PipelineRefusal::ForeignDevice)
            ));
        }
        assert!(buffer.receive(&device, 1, &[0; 15]).is_err());
        assert_eq!(buffer.tensor.storage_bytes(), before);
    }

    #[test]
    fn activation_contract_requires_exact_bounded_f32_rows() {
        let contract = ActivationContract::new(2, 2560).unwrap();
        assert_eq!(contract.bytes(1).unwrap(), 10240);
        assert_eq!(contract.bytes(2).unwrap(), 20480);
        assert!(contract.validate(2, Element::f32(), 20480).is_ok());
        for rows in [0, 3, u64::MAX] {
            assert!(contract.bytes(rows).is_err());
        }
        assert!(contract.validate(1, Element::f16(), 10240).is_err());
        for bytes in [0, 10239, 10241, 20480] {
            assert!(contract.validate(1, Element::f32(), bytes).is_err());
        }
        assert!(ActivationContract::new(1, 0).is_err());
        assert!(ActivationContract::new(2, u64::MAX).is_err());
        assert!(ActivationContract::new(3, 2560).is_err());
        assert!(ActivationContract::new(1, 2560).unwrap().bytes(2).is_err());
    }
}
