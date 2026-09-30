//! Request residency, resume states, and reclaim.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpenRequirements {
    target_banks: usize,
    head_banks: usize,
}

impl OpenRequirements {
    pub fn target_banks(self) -> usize {
        self.target_banks
    }
    pub fn head_banks(self) -> usize {
        self.head_banks
    }
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

    /// Make a request with installed input resident: fresh state at zero, or
    /// `from`'s state at its position. Features `from` holds for spans of
    /// this input straddling that position are adopted; every span ending
    /// after it that still lacks features is returned as an encode. Opening
    /// into free banks changes no bindings; growing one orphans a queued
    /// lookahead first.
    pub fn open_state(
        &mut self,
        bindings: &mut StateBindings<F>,
        request: RequestId,
        from: Option<&ResumeState>,
    ) -> Result<Vec<Operation>, DomainError> {
        self.orphan_lookahead(bindings)?;
        let installed = self
            .input
            .get(&request)
            .ok_or_else(|| DomainError::invariant("request has no installed input"))?;
        if installed.resident || self.target.contains_key(&request) || self.head.contains_key(&request)
        {
            return Err(DomainError::invariant(format!(
                "request {} is already resident",
                request.0
            )));
        }
        let (target, head) = match from {
            Some(from) => self.fork_resume_state(from)?,
            None => self.fresh_state(bindings)?,
        };
        let position = target.position();
        self.target.insert(request, target);
        if let Some(head) = head {
            self.head.insert(request, head);
        }
        let installed = self.input.get_mut(&request).expect("input checked above");
        installed.resident = true;
        let mut encodes = Vec::new();
        for span in installed.input.layout().spans() {
            if span.end <= position {
                continue;
            }
            let slot = installed
                .images
                .get_mut(&span.identity)
                .expect("installed input has a slot for every span");
            if slot.features.is_none() && span.start < position {
                slot.features = from.and_then(|from| from.features.get(&span.identity).cloned());
            }
            if slot.features.is_none() {
                encodes.push(Operation::Encode {
                    request,
                    image: slot.image.clone(),
                });
            }
        }
        Ok(encodes)
    }

    fn fresh_state(
        &mut self,
        bindings: &mut StateBindings<F>,
    ) -> Result<(SequenceState, Option<SequenceState>), DomainError> {
        self.provision_open(bindings)?;
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
        Ok((target, head))
    }

    fn fork_resume_state(
        &self,
        from: &ResumeState,
    ) -> Result<(SequenceState, Option<SequenceState>), DomainError> {
        let target = from.target.fork();
        if !target.belongs_to(&self.target_store) || from.head.is_some() != self.head_store.is_some()
        {
            return Err(DomainError::invariant(
                "resume state differs from domain state arenas",
            ));
        }
        let head = from.head.as_ref().map(StateCheckpoint::fork);
        if head.as_ref().is_some_and(|state| {
            !self
                .head_store
                .as_ref()
                .is_some_and(|store| state.belongs_to(store))
        }) {
            return Err(DomainError::invariant(
                "resume state head belongs to another state arena",
            ));
        }
        Ok((target, head))
    }

    /// The request's reconciled state as a resume state: both lanes and the
    /// features of spans straddling its position.
    pub fn resume_state(&self, request: RequestId) -> Result<ResumeState, String> {
        let target = self
            .target
            .get(&request)
            .ok_or_else(|| format!("request {} has in-flight work or is not open", request.0))?
            .checkpoint();
        if self.head_store.is_some() && !self.head.contains_key(&request) {
            return Err("request has unresolved head work".into());
        }
        let head = self.head.get(&request).map(SequenceState::checkpoint);
        let installed = self
            .input
            .get(&request)
            .ok_or("resident request has no installed input")?;
        let position = target.position();
        let features = installed
            .input
            .layout()
            .spans()
            .iter()
            .filter(|span| span.start < position && position < span.end)
            .map(|span| {
                let features = installed
                    .images
                    .get(&span.identity)
                    .and_then(|slot| slot.features.clone())
                    .ok_or("resident rows condition on an unencoded image")?;
                Ok((span.identity.clone(), features))
            })
            .collect::<Result<_, String>>()?;
        Ok(ResumeState {
            target,
            head,
            features,
        })
    }

    /// Drop a request's input and any numerical state. A resident request's
    /// state must not be in flight.
    pub fn close(&mut self, request: RequestId) -> Result<(), String> {
        let installed = self.input.get(&request).ok_or("request is not open")?;
        if installed.resident
            && (!self.target.contains_key(&request)
                || (self.head_store.is_some() && !self.head.contains_key(&request)))
        {
            return Err("request has unresolved work".into());
        }
        self.head.remove(&request);
        self.input.remove(&request);
        self.target.remove(&request);
        Ok(())
    }

    /// State bytes released by evicting exactly these requests, priced as a
    /// set: rows and banks shared with anything outside it are not counted.
    pub fn reclaimable(&self, requests: &[RequestId]) -> Result<u64, String> {
        let target = requests
            .iter()
            .map(|request| {
                self.target.get(request).map(Holder::State).ok_or_else(|| {
                    "reclamation request has unresolved or absent target state".to_owned()
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut bytes = self
            .target_store
            .exclusive_bytes(&target)
            .map_err(|error| error.to_string())?;
        if let Some(store) = &self.head_store {
            let head = requests
                .iter()
                .map(|request| {
                    self.head.get(request).map(Holder::State).ok_or_else(|| {
                        "reclamation request has unresolved or absent head state".to_owned()
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            bytes = bytes
                .checked_add(
                    store
                        .exclusive_bytes(&head)
                        .map_err(|error| error.to_string())?,
                )
                .ok_or("reclaim byte count overflow")?;
        }
        Ok(bytes)
    }

    /// Evict these requests: release their numerical state and encoded
    /// features. Their input stays installed for when they are resident
    /// again.
    pub fn release_state(&mut self, requests: &[RequestId]) -> Result<u64, String> {
        let bytes = self.reclaimable(requests)?;
        for request in requests {
            let installed = self
                .input
                .get_mut(request)
                .ok_or("evicted request has no installed input")?;
            installed.resident = false;
            for slot in installed.images.values_mut() {
                slot.features = None;
            }
            self.target.remove(request);
            self.head.remove(request);
        }
        Ok(bytes)
    }

    /// Release every slab of a store no sequence or checkpoint owns. That
    /// changes bindings, so a queued lookahead is orphaned first.
    pub fn reclaim_idle(&mut self, bindings: &mut StateBindings<F>) -> Result<u64, String> {
        self.orphan_lookahead(bindings)
            .map_err(|error| error.to_string())?;
        let mut bytes = u64::try_from(
            bindings
                .target
                .release_idle()
                .map_err(|error| error.to_string())?,
        )
        .map_err(|_| "target idle bytes exceed u64")?;
        if let Some(store) = &mut bindings.head {
            bytes = bytes
                .checked_add(
                    u64::try_from(store.release_idle().map_err(|error| error.to_string())?)
                        .map_err(|_| "head idle bytes exceed u64")?,
                )
                .ok_or("idle reclaim byte count overflow")?;
        }
        Ok(bytes)
    }
}
