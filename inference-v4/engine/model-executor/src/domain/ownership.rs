//! Accepted request state, checkpoints, and reclaim.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpenRequirements {
    target_banks: usize,
    head_banks: usize,
}

pub struct OpenReservation {
    request: RequestId,
    target: SequenceState,
    head: Option<SequenceState>,
}

impl<F: ProgramFamily> ExecutorDomain<F> {
    pub fn open_requirements(&self) -> OpenRequirements {
        OpenRequirements {
            target_banks: 1,
            head_banks: usize::from(self.head_store.is_some()),
        }
    }

    pub fn can_open(&self, requirements: OpenRequirements) -> Result<(), CapacityError> {
        let target = self.target_store.available_banks();
        if target < requirements.target_banks {
            return Err(CapacityError {
                resource: ResourceKind::RecurrentBanks,
                required: requirements.target_banks as u64,
                available: target as u64,
            });
        }
        if let Some(store) = &self.head_store {
            let head = store.available_banks();
            if head < requirements.head_banks {
                return Err(CapacityError {
                    resource: ResourceKind::RecurrentBanks,
                    required: requirements.head_banks as u64,
                    available: head as u64,
                });
            }
        }
        Ok(())
    }

    pub fn reserve_open(&mut self, request: RequestId) -> Result<OpenReservation, DomainError> {
        self.healthy()?;
        self.ensure_closed(request)?;
        let requirements = self.open_requirements();
        self.can_open(requirements).map_err(DomainError::Capacity)?;
        let target = self.target_store.create().map_err(|error| {
            DomainError::invariant(format!(
                "target admission capacity changed after availability check: {error}"
            ))
        })?;
        let head = self
            .head_store
            .as_ref()
            .map(|store| {
                store.create().map_err(|error| {
                    DomainError::invariant(format!(
                        "head admission capacity changed after availability check: {error}"
                    ))
                })
            })
            .transpose()?;
        Ok(OpenReservation {
            request,
            target,
            head,
        })
    }

    pub fn open_reserved(&mut self, reservation: OpenReservation) -> Result<(), DomainError> {
        self.healthy()?;
        self.ensure_closed(reservation.request)?;
        self.target.insert(reservation.request, reservation.target);
        if let Some(state) = reservation.head {
            self.head.insert(reservation.request, state);
        }
        Ok(())
    }

    pub fn open(&mut self, request: RequestId) -> Result<(), DomainError> {
        let reservation = self.reserve_open(request)?;
        self.open_reserved(reservation)
    }

    fn ensure_closed(&self, request: RequestId) -> Result<(), DomainError> {
        if self.target.contains_key(&request)
            || self.head.contains_key(&request)
            || self.head_pending.contains_key(&request)
            || self.input.contains_key(&request)
            || self.repairs.contains_key(&request)
        {
            return Err(format!("request {} is already open", request.0).into());
        }
        Ok(())
    }

    pub fn close(&mut self, request: RequestId) -> Result<(), String> {
        if self.head_pending.contains_key(&request)
            || self.repairs.contains_key(&request)
            || !self.target.contains_key(&request)
            || (self.head_store.is_some() && !self.head.contains_key(&request))
        {
            return Err("request has unresolved work or is not open".into());
        }
        self.head.remove(&request);
        self.input.remove(&request);
        self.target.remove(&request);
        Ok(())
    }

    pub fn checkpoint(&self, request: RequestId) -> Result<DomainCheckpoint, String> {
        let target = self
            .target
            .get(&request)
            .ok_or_else(|| format!("request {} has in-flight work or is not open", request.0))?
            .checkpoint();
        if self.head_store.is_some() && !self.head.contains_key(&request) {
            return Err("request has unresolved head work".into());
        }
        let head = self.head.get(&request).map(SequenceState::checkpoint);
        Ok(DomainCheckpoint {
            target,
            head,
            input: self.input.get(&request).cloned(),
        })
    }

    pub fn restore(
        &mut self,
        request: RequestId,
        checkpoint: &DomainCheckpoint,
    ) -> Result<(), DomainError> {
        let current = self
            .target
            .get(&request)
            .ok_or_else(|| "request is not idle".to_owned())?;
        let replacement = checkpoint.target.fork();
        if !replacement.belongs_to(&self.target_store)
            || replacement.position() > current.position()
        {
            return Err(
                "checkpoint belongs to another state arena or is ahead of the request".into(),
            );
        }
        let head = checkpoint.head.as_ref().map(StateCheckpoint::fork);
        if checkpoint.head.is_some() != self.head_store.is_some()
            || head.as_ref().is_some_and(|state| {
                !self
                    .head_store
                    .as_ref()
                    .is_some_and(|store| state.belongs_to(store))
            })
        {
            return Err("checkpoint head belongs to another state arena".into());
        }
        self.target.insert(request, replacement);
        if let Some(head) = head {
            self.head.insert(request, head);
        }
        if let Some(input) = &checkpoint.input {
            self.input.insert(request, input.clone());
        }
        Ok(())
    }

    pub fn open_checkpoint(
        &mut self,
        request: RequestId,
        checkpoint: &DomainCheckpoint,
    ) -> Result<(), DomainError> {
        self.healthy()?;
        if self.target.contains_key(&request)
            || self.head.contains_key(&request)
            || self.head_pending.contains_key(&request)
            || self.input.contains_key(&request)
            || self.repairs.contains_key(&request)
        {
            return Err("checkpoint request is already open".into());
        }
        let target = checkpoint.target.fork();
        if !target.belongs_to(&self.target_store)
            || checkpoint.head.is_some() != self.head_store.is_some()
        {
            return Err("checkpoint differs from domain state arenas".into());
        }
        let head = checkpoint.head.as_ref().map(StateCheckpoint::fork);
        if head.as_ref().is_some_and(|state| {
            !self
                .head_store
                .as_ref()
                .is_some_and(|store| state.belongs_to(store))
        }) {
            return Err("checkpoint head belongs to another state arena".into());
        }
        self.target.insert(request, target);
        if let Some(head) = head {
            self.head.insert(request, head);
        }
        if let Some(input) = &checkpoint.input {
            self.input.insert(request, input.clone());
        }
        Ok(())
    }

    pub fn reclaimable(&self, requests: &[RequestId]) -> Result<u64, String> {
        let target = requests
            .iter()
            .map(|request| {
                self.target.get(request).ok_or_else(|| {
                    "reclamation request has unresolved or absent target state".to_owned()
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut bytes = u64::try_from(
            self.target_store
                .reclaimable(&target)
                .map_err(|error| error.to_string())?,
        )
        .map_err(|_| "target reclaim bytes exceed u64")?;
        if let Some(store) = &self.head_store {
            let head = requests
                .iter()
                .map(|request| {
                    self.head.get(request).ok_or_else(|| {
                        "reclamation request has unresolved or absent head state".to_owned()
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            bytes = bytes
                .checked_add(
                    u64::try_from(
                        store
                            .reclaimable(&head)
                            .map_err(|error| error.to_string())?,
                    )
                    .map_err(|_| "head reclaim bytes exceed u64")?,
                )
                .ok_or("reclaim byte count overflow")?;
        }
        Ok(bytes)
    }

    pub fn evict(&mut self, requests: &[RequestId]) -> Result<u64, String> {
        let bytes = self.reclaimable(requests)?;
        for request in requests {
            self.target.remove(request);
            self.head.remove(request);
            self.input.remove(request);
        }
        Ok(bytes)
    }

    pub fn reclaim_idle(&mut self) -> Result<u64, String> {
        let mut bytes = u64::try_from(
            self.target_store
                .release_idle()
                .map_err(|error| error.to_string())?,
        )
        .map_err(|_| "target idle bytes exceed u64")?;
        if let Some(store) = &self.head_store {
            bytes = bytes
                .checked_add(
                    u64::try_from(store.release_idle().map_err(|error| error.to_string())?)
                        .map_err(|_| "head idle bytes exceed u64")?,
                )
                .ok_or("idle reclaim byte count overflow")?;
        }
        Ok(bytes)
    }

    pub fn fork(&mut self, source: RequestId, destination: RequestId) -> Result<(), String> {
        if self.target.contains_key(&destination)
            || self.head.contains_key(&destination)
            || self.head_pending.contains_key(&destination)
            || self.input.contains_key(&destination)
            || self.repairs.contains_key(&destination)
        {
            return Err("fork destination is already open".into());
        }
        let state = self
            .target
            .get(&source)
            .ok_or_else(|| "fork source is not idle".to_owned())?
            .checkpoint()
            .fork();
        if self.head_store.is_some() && !self.head.contains_key(&source) {
            return Err("fork source has unresolved head work".into());
        }
        self.target.insert(destination, state);
        if let Some(state) = self
            .head
            .get(&source)
            .map(|state| state.checkpoint().fork())
        {
            self.head.insert(destination, state);
        }
        if let Some(input) = self.input.get(&source).cloned() {
            self.input.insert(destination, input);
        }
        Ok(())
    }
}
