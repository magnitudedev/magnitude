use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestId(pub u64);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Prefill,
    Decode,
}
#[derive(Clone, Debug)]
pub struct Limits {
    pub max_requests: usize,
    pub max_batch: usize,
    pub prefill_tokens: usize,
    pub decode_tokens: usize,
    pub decode_share: f64,
    pub locality_seconds: f64,
}
impl Limits {
    fn validate(&self) -> Result<(), String> {
        if self.max_requests == 0
            || self.max_batch == 0
            || self.prefill_tokens == 0
            || self.decode_tokens == 0
            || self.decode_tokens > 256
            || !self.decode_share.is_finite()
            || self.decode_share <= 0.0
            || self.decode_share >= 1.0
            || !self.locality_seconds.is_finite()
            || self.locality_seconds < 0.0
            || !(self.locality_seconds * 1e9 * (f64::from(u32::MAX) + 2.0)).is_finite()
        {
            return Err("invalid service limits".into());
        }
        Ok(())
    }
}
#[derive(Clone, Debug)]
pub struct Operation {
    pub identity: RequestId,
    pub phase: Phase,
    pub active: bool,
    pub resident: bool,
    pub waiting_since_ns: u64,
    pub service_ns: u64,
    pub preemption_debt: u32,
}
/// A selection is consumed when its preparation-through-completion cost is charged.
#[derive(Debug)]
pub struct Selection {
    phase: Phase,
    requests: Vec<RequestId>,
    contended: bool,
}
impl Selection {
    pub fn phase(&self) -> Phase {
        self.phase
    }
    pub fn requests(&self) -> &[RequestId] {
        &self.requests
    }
    pub fn contended(&self) -> bool {
        self.contended
    }
    /// Capacity preparation may drop trailing members without changing phase.
    pub fn truncate(&mut self, count: usize) -> Result<(), String> {
        if count == 0 || count > self.requests.len() {
            return Err("invalid prepared batch size".into());
        }
        self.requests.truncate(count);
        Ok(())
    }
}
pub struct Scheduler {
    limits: Limits,
    decode_debt_ns: f64,
    contended: bool,
    completed_service_ns: u128,
}
impl Scheduler {
    pub fn new(limits: Limits) -> Result<Self, String> {
        limits.validate()?;
        Ok(Self {
            limits,
            decode_debt_ns: 0.0,
            contended: false,
            completed_service_ns: 0,
        })
    }
    pub fn limits(&self) -> &Limits {
        &self.limits
    }
    pub fn completed_service_ns(&self) -> u128 {
        self.completed_service_ns
    }
    pub fn select(
        &mut self,
        candidates: &[Operation],
        now_ns: u64,
    ) -> Result<Option<Selection>, String> {
        let mut identities = HashSet::new();
        if candidates
            .iter()
            .any(|c| c.waiting_since_ns > now_ns || !identities.insert(c.identity))
        {
            return Err("scheduler candidates require unique IDs and valid waiting times".into());
        }
        if candidates.is_empty() {
            self.contended = false;
            self.decode_debt_ns = 0.0;
            return Ok(None);
        }
        let decoding = candidates.iter().any(|c| c.phase == Phase::Decode);
        let prefill = candidates.iter().any(|c| c.phase == Phase::Prefill);
        let contended = decoding && prefill;
        let first = contended && !self.contended;
        if !contended || first {
            self.decode_debt_ns = 0.0;
        }
        self.contended = contended;
        let phase = if decoding && (!prefill || first || self.decode_debt_ns > 0.0) {
            Phase::Decode
        } else {
            Phase::Prefill
        };
        let priority = |c: &Operation| {
            (now_ns - c.waiting_since_ns) as f64
                + self.limits.locality_seconds
                    * 1e9
                    * (u64::from(c.active) + u64::from(c.resident) + u64::from(c.preemption_debt))
                        as f64
        };
        let mut eligible = candidates
            .iter()
            .filter(|c| c.phase == phase)
            .collect::<Vec<_>>();
        eligible.sort_by(|a, b| {
            priority(b)
                .total_cmp(&priority(a))
                .then_with(|| a.service_ns.cmp(&b.service_ns))
                .then_with(|| a.identity.cmp(&b.identity))
        });
        let requests = eligible
            .into_iter()
            .take(self.limits.max_batch)
            .map(|c| c.identity)
            .collect();
        Ok(Some(Selection {
            phase,
            requests,
            contended,
        }))
    }
    pub fn completed(&mut self, selection: Selection, elapsed_ns: u64) {
        self.completed_service_ns += u128::from(elapsed_ns);
        if selection.contended {
            match selection.phase {
                Phase::Decode => {
                    self.decode_debt_ns = (self.decode_debt_ns - elapsed_ns as f64).max(0.0)
                }
                Phase::Prefill => {
                    self.decode_debt_ns += elapsed_ns as f64 * self.limits.decode_share
                        / (1.0 - self.limits.decode_share)
                }
            }
        }
    }
}

/// Candidate eligibility (live, resident, unsubmitted, unprotected, unselected)
/// belongs to the service. Byte prices come from the model for actual ownership.
pub struct Victim {
    pub identity: RequestId,
    pub output_blocked: bool,
    pub preemption_debt: u32,
    pub exclusive_bytes: u64,
    pub replay_tokens: u64,
    pub service_ns: u64,
}
/// Ordered candidates only. The service must price growing *sets* again because
/// shared allocations may become reclaimable only after several owners close.
pub fn order_victims(victims: &mut [Victim]) {
    victims.sort_by(|a, b| {
        b.output_blocked
            .cmp(&a.output_blocked)
            .then_with(|| a.preemption_debt.cmp(&b.preemption_debt))
            .then_with(|| {
                let left = u128::from(a.exclusive_bytes) * u128::from(b.replay_tokens.max(1));
                let right = u128::from(b.exclusive_bytes) * u128::from(a.replay_tokens.max(1));
                right.cmp(&left)
            })
            .then_with(|| a.service_ns.cmp(&b.service_ns))
            .then_with(|| a.identity.cmp(&b.identity))
    });
}

/// Retry only when an event capable of changing capacity advances the epoch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct CapacityEpoch(u64);
impl CapacityEpoch {
    pub fn advance(&mut self) -> Result<(), String> {
        self.0 = self.0.checked_add(1).ok_or("capacity epoch exhausted")?;
        Ok(())
    }
    pub fn changed_since(self, blocked: Self) -> bool {
        self != blocked
    }
}
