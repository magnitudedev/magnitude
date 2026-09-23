use magnitude_artifacts::{InputLayout, PackageIdentity, TokenId};
use magnitude_generation::MethodCheckpoint;
use magnitude_model_contracts::{PreparedModelInput, TokenPlan};
use magnitude_model_executor::ResourcePlan;
use magnitude_model_state::CodecIdentity;
use std::collections::VecDeque;
use std::sync::Arc;

pub const MIN_RETENTION_HIT: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetentionCapacity {
    pub max_bytes: u64,
    pub max_entries: usize,
}

impl RetentionCapacity {
    pub const fn disabled() -> Self {
        Self {
            max_bytes: 0,
            max_entries: 0,
        }
    }

    pub fn from_resource_plan(plan: &ResourcePlan) -> Self {
        Self {
            max_bytes: plan.retention_budget_bytes(),
            max_entries: plan.capacity().retention_entries,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TokenizerIdentity(Arc<str>);

impl TokenizerIdentity {
    pub fn new(value: impl Into<Arc<str>>) -> Result<Self, String> {
        let value = value.into();
        if value.is_empty() {
            return Err("tokenizer identity must not be empty".into());
        }
        Ok(Self(value))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RetentionKey {
    artifact: PackageIdentity,
    tokenizer: TokenizerIdentity,
    codec: CodecIdentity,
}

impl RetentionKey {
    pub fn new(
        artifact: PackageIdentity,
        tokenizer: TokenizerIdentity,
        codec: CodecIdentity,
    ) -> Self {
        Self {
            artifact,
            tokenizer,
            codec,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RetentionRequest {
    pub key: RetentionKey,
    pub tokens: Vec<TokenId>,
    pub conditioning: Vec<String>,
    layout: InputLayout,
}

impl RetentionRequest {
    pub fn new(key: RetentionKey, input: &PreparedModelInput) -> Result<Self, String> {
        Self::from_parts(key, input.tokens(), input.layout())
    }

    pub fn from_token_plan(key: RetentionKey, plan: &TokenPlan) -> Result<Self, String> {
        Self::from_parts(key, plan.tokens(), plan.layout())
    }

    fn from_parts(
        key: RetentionKey,
        tokens: &[TokenId],
        layout: &InputLayout,
    ) -> Result<Self, String> {
        Ok(Self {
            key,
            tokens: tokens.to_vec(),
            conditioning: layout
                .spans()
                .iter()
                .map(|span| span.identity.clone())
                .collect(),
            layout: layout.clone(),
        })
    }

    pub fn exact_boundary(&self, position: usize) -> bool {
        self.layout.boundary(position)
    }

    pub fn conditioning_at(&self, position: usize) -> Result<&[String], String> {
        if !self.exact_boundary(position) {
            return Err("retention position is not an exact input boundary".into());
        }
        let count = self
            .layout
            .spans()
            .iter()
            .take_while(|span| span.start < position)
            .count();
        Ok(&self.conditioning[..count])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetentionHit {
    id: u64,
    position: usize,
}

impl RetentionHit {
    pub const fn id(self) -> u64 {
        self.id
    }

    pub const fn position(self) -> usize {
        self.position
    }
}

pub struct RetainedEntry<C> {
    id: u64,
    key: RetentionKey,
    checkpoint: C,
    method: MethodCheckpoint,
    tokens: Vec<TokenId>,
    conditioning: Vec<String>,
    position: usize,
    last_use: u64,
    submitted: usize,
    charged: u64,
}

impl<C> RetainedEntry<C> {
    pub const fn checkpoint(&self) -> &C {
        &self.checkpoint
    }

    pub const fn method(&self) -> &MethodCheckpoint {
        &self.method
    }

    pub const fn position(&self) -> usize {
        self.position
    }

    pub const fn charged_bytes(&self) -> u64 {
        self.charged
    }
}

/// Thread-confined exact-prefix checkpoint index. Checkpoints are owned by the
/// index; a hit only borrows the entry while the executor forks its numerical
/// state and the fresh generation restores the owned method checkpoint.
pub struct Retention<C> {
    capacity: RetentionCapacity,
    charged: u64,
    clock: u64,
    next_id: u64,
    entries: VecDeque<RetainedEntry<C>>,
}

impl<C> Retention<C> {
    pub const fn new(capacity: RetentionCapacity) -> Self {
        Self {
            capacity,
            charged: 0,
            clock: 0,
            next_id: 1,
            entries: VecDeque::new(),
        }
    }

    pub const fn budget_bytes(&self) -> u64 {
        self.capacity.max_bytes
    }

    pub const fn max_entries(&self) -> usize {
        self.capacity.max_entries
    }

    pub const fn charged_bytes(&self) -> u64 {
        self.charged
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn entry(&self, hit: RetentionHit) -> Result<&RetainedEntry<C>, String> {
        self.entries
            .iter()
            .find(|entry| entry.id == hit.id)
            .ok_or_else(|| "retention hit is no longer resident".to_owned())
    }

    pub fn lookup(&mut self, request: &RetentionRequest) -> Result<Option<RetentionHit>, String> {
        let mut best = None;
        for (index, entry) in self.entries.iter().enumerate() {
            if entry.submitted != 0
                || entry.key != request.key
                || entry.position < MIN_RETENTION_HIT
                || entry.tokens.len() != entry.position
                || !request.tokens.starts_with(&entry.tokens)
                || entry.conditioning != request.conditioning_at(entry.position)?
            {
                continue;
            }
            if best.is_none_or(|(_, position, last_use)| {
                entry.position > position
                    || (entry.position == position && entry.last_use > last_use)
            }) {
                best = Some((index, entry.position, entry.last_use));
            }
        }
        let Some((index, position, _)) = best else {
            return Ok(None);
        };
        let last_use = self.tick()?;
        let entry = &mut self.entries[index];
        entry.last_use = last_use;
        Ok(Some(RetentionHit {
            id: entry.id,
            position,
        }))
    }

    pub fn retain(
        &mut self,
        request: &RetentionRequest,
        tokens: Vec<TokenId>,
        checkpoint: C,
        checkpoint_bytes: u64,
        method: MethodCheckpoint,
    ) -> Result<bool, String> {
        let position = tokens.len();
        if position == 0
            || !request.exact_boundary(position)
            || !tokens.starts_with(&request.tokens)
        {
            return Err("retained state is not an exact extension of its interpreted input".into());
        }
        let conditioning = request.conditioning_at(position)?.to_vec();
        let charged = checkpoint_bytes
            .checked_add(method.retained_bytes())
            .ok_or("retained checkpoint byte count overflow")?;
        if self.capacity.max_bytes == 0
            || self.capacity.max_entries == 0
            || charged > self.capacity.max_bytes
        {
            return Ok(false);
        }
        loop {
            let required_bytes = self
                .charged
                .checked_add(charged)
                .ok_or("retention budget charge overflow")?
                .saturating_sub(self.capacity.max_bytes);
            let requires_entry = self.entries.len() >= self.capacity.max_entries;
            if required_bytes == 0 && !requires_entry {
                break;
            }
            if self.evict_one()?.is_none() {
                return Ok(false);
            }
        }
        let last_use = self.tick()?;
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or("retention entry identity exhausted")?;
        self.charged = self
            .charged
            .checked_add(charged)
            .ok_or("retention budget charge overflow")?;
        self.entries.push_back(RetainedEntry {
            id,
            key: request.key.clone(),
            checkpoint,
            method,
            tokens,
            conditioning,
            position,
            last_use,
            submitted: 0,
            charged,
        });
        Ok(true)
    }

    pub fn begin_submitted(&mut self, id: u64) -> Result<(), String> {
        let entry = self
            .entries
            .iter_mut()
            .find(|entry| entry.id == id)
            .ok_or("retention entry is no longer resident")?;
        entry.submitted = entry
            .submitted
            .checked_add(1)
            .ok_or("retention submitted-use count exhausted")?;
        Ok(())
    }

    pub fn begin_submitted_if_resident(&mut self, id: u64) -> Result<bool, String> {
        if self.entries.iter().all(|entry| entry.id != id) {
            return Ok(false);
        }
        self.begin_submitted(id)?;
        Ok(true)
    }

    pub fn end_submitted(&mut self, id: u64) -> Result<(), String> {
        let entry = self
            .entries
            .iter_mut()
            .find(|entry| entry.id == id)
            .ok_or("retention entry is no longer resident")?;
        entry.submitted = entry
            .submitted
            .checked_sub(1)
            .ok_or("retention entry has no outstanding submitted use")?;
        Ok(())
    }

    /// Capacity-ladder rung zero: release least-recently-used retained entries
    /// before asking live requests or executor arenas for bytes.
    pub fn evict_bytes(&mut self, required: u64) -> Result<u64, String> {
        let mut released = 0_u64;
        while released < required {
            let Some(bytes) = self.evict_one()? else {
                break;
            };
            released = released.saturating_add(bytes);
        }
        Ok(released)
    }

    fn evict_one(&mut self) -> Result<Option<u64>, String> {
        let candidate = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.submitted == 0)
            .min_by_key(|(_, entry)| entry.last_use)
            .map(|(index, _)| index);
        let Some(index) = candidate else {
            return Ok(None);
        };
        let entry = self.entries.remove(index).unwrap();
        self.charged = self
            .charged
            .checked_sub(entry.charged)
            .ok_or("retention charge accounting underflow")?;
        Ok(Some(entry.charged))
    }

    fn tick(&mut self) -> Result<u64, String> {
        self.clock = self
            .clock
            .checked_add(1)
            .ok_or("retention LRU clock exhausted")?;
        Ok(self.clock)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_artifacts::{BoundaryRule, InputLayout, InputSpan};

    const fn capacity(max_bytes: u64, max_entries: usize) -> RetentionCapacity {
        RetentionCapacity {
            max_bytes,
            max_entries,
        }
    }

    fn conditioned_plan() -> TokenPlan {
        TokenPlan::new(
            (0..10).map(TokenId).collect(),
            InputLayout::new(
                10,
                vec![
                    InputSpan {
                        start: 2,
                        end: 5,
                        identity: "first".into(),
                        boundaries: BoundaryRule::Causal,
                        language_history: false,
                    },
                    InputSpan {
                        start: 7,
                        end: 9,
                        identity: "second".into(),
                        boundaries: BoundaryRule::Indivisible,
                        language_history: false,
                    },
                ],
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn key() -> RetentionKey {
        RetentionKey::new(
            PackageIdentity {
                target: magnitude_artifacts::ArtifactIdentity([1; 32]),
                projector: None,
            },
            TokenizerIdentity::new("tokenizer").unwrap(),
            CodecIdentity::new("codec").unwrap(),
        )
    }

    #[test]
    fn request_owns_interpreted_tokens_and_conditioning_identities() {
        let plan = conditioned_plan();
        let request = RetentionRequest::from_token_plan(key(), &plan).unwrap();
        assert_eq!(request.tokens, plan.tokens());
        assert_eq!(request.conditioning, ["first", "second"]);
        assert_eq!(request.conditioning_at(2).unwrap(), &[] as &[String]);
        assert_eq!(request.conditioning_at(3).unwrap(), ["first"]);
        assert_eq!(request.conditioning_at(7).unwrap(), ["first"]);
        assert_eq!(request.conditioning_at(9).unwrap(), ["first", "second"]);
        assert!(request.conditioning_at(8).is_err());
    }

    #[test]
    fn request_rejects_unidentified_retention_domains() {
        let plan = TokenPlan::new(vec![TokenId(1)], InputLayout::new(1, vec![]).unwrap()).unwrap();
        assert!(TokenizerIdentity::new("").is_err());
        assert!(CodecIdentity::new("").is_err());
        assert!(RetentionRequest::from_token_plan(key(), &plan).is_ok());
    }

    fn text_request(count: usize, artifact: &str) -> RetentionRequest {
        RetentionRequest::from_token_plan(
            RetentionKey::new(
                PackageIdentity {
                    target: magnitude_artifacts::ArtifactIdentity(
                        [artifact.as_bytes().first().copied().unwrap_or(0); 32],
                    ),
                    projector: None,
                },
                TokenizerIdentity::new("tokenizer").unwrap(),
                CodecIdentity::new("codec").unwrap(),
            ),
            &TokenPlan::new(
                (0..count as u32).map(TokenId).collect(),
                InputLayout::new(count, vec![]).unwrap(),
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn index_selects_the_longest_exact_prefix_and_enforces_minimum_hit() {
        let mut retention = Retention::new(capacity(100, 8));
        let short = text_request(63, "artifact");
        retention
            .retain(
                &short,
                short.tokens.clone(),
                "short",
                1,
                MethodCheckpoint::Plain,
            )
            .unwrap();
        let middle = text_request(64, "artifact");
        retention
            .retain(
                &middle,
                middle.tokens.clone(),
                "middle",
                1,
                MethodCheckpoint::Plain,
            )
            .unwrap();
        let long = text_request(80, "artifact");
        retention
            .retain(
                &long,
                long.tokens.clone(),
                "long",
                1,
                MethodCheckpoint::Plain,
            )
            .unwrap();

        let target = text_request(96, "artifact");
        let hit = retention.lookup(&target).unwrap().unwrap();
        assert_eq!(hit.position(), 80);
        assert_eq!(retention.entry(hit).unwrap().checkpoint(), &"long");
        assert!(retention
            .lookup(&text_request(63, "artifact"))
            .unwrap()
            .is_none());
        assert!(retention
            .lookup(&text_request(96, "other"))
            .unwrap()
            .is_none());
    }

    #[test]
    fn terminal_suffix_is_an_exact_boundary_and_can_seed_a_longer_prompt() {
        let source = text_request(64, "artifact");
        let mut terminal = source.tokens.clone();
        terminal.extend((64..72).map(TokenId));
        let mut retention = Retention::new(capacity(100, 8));
        assert!(retention
            .retain(
                &source,
                terminal.clone(),
                "terminal",
                1,
                MethodCheckpoint::Plain,
            )
            .unwrap());

        let target = text_request(80, "artifact");
        let hit = retention.lookup(&target).unwrap().unwrap();
        assert_eq!(hit.position(), 72);
        assert_eq!(retention.entry(hit).unwrap().checkpoint(), &"terminal");
    }

    #[test]
    fn index_rejects_conditioning_identity_mismatch() {
        let conditioned = |identity: &str| {
            let plan = TokenPlan::new(
                (0..64).map(TokenId).collect(),
                InputLayout::new(
                    64,
                    vec![InputSpan {
                        start: 2,
                        end: 5,
                        identity: identity.into(),
                        boundaries: BoundaryRule::Causal,
                        language_history: false,
                    }],
                )
                .unwrap(),
            )
            .unwrap();
            RetentionRequest::from_token_plan(key(), &plan).unwrap()
        };
        let source = conditioned("source");
        let mut retention = Retention::new(capacity(100, 8));
        retention
            .retain(
                &source,
                source.tokens.clone(),
                (),
                1,
                MethodCheckpoint::Plain,
            )
            .unwrap();

        let changed = conditioned("changed");
        assert!(retention.lookup(&changed).unwrap().is_none());
    }

    #[test]
    fn budget_evicts_oldest_use_and_never_evicts_or_hits_submitted_entries() {
        let mut retention = Retention::new(capacity(10, 8));
        let first = text_request(64, "first");
        let second = text_request(64, "second");
        let third = text_request(64, "third");
        retention
            .retain(&first, first.tokens.clone(), 1, 5, MethodCheckpoint::Plain)
            .unwrap();
        retention
            .retain(
                &second,
                second.tokens.clone(),
                2,
                5,
                MethodCheckpoint::Plain,
            )
            .unwrap();
        let first_hit = retention.lookup(&first).unwrap().unwrap();
        retention.begin_submitted(first_hit.id()).unwrap();
        assert!(retention.lookup(&first).unwrap().is_none());
        retention
            .retain(&third, third.tokens.clone(), 3, 5, MethodCheckpoint::Plain)
            .unwrap();
        assert_eq!(retention.charged_bytes(), 10);
        assert!(retention.lookup(&second).unwrap().is_none());
        assert!(retention.lookup(&third).unwrap().is_some());
        retention.end_submitted(first_hit.id()).unwrap();
        assert!(retention.lookup(&first).unwrap().is_some());
    }

    #[test]
    fn numerical_entry_capacity_evicts_even_when_byte_budget_has_room() {
        let mut retention = Retention::new(capacity(1_000, 2));
        let first = text_request(64, "first");
        let second = text_request(64, "second");
        let third = text_request(64, "third");
        assert!(retention
            .retain(&first, first.tokens.clone(), 1, 1, MethodCheckpoint::Plain)
            .unwrap());
        assert!(retention
            .retain(
                &second,
                second.tokens.clone(),
                2,
                1,
                MethodCheckpoint::Plain
            )
            .unwrap());
        assert!(retention
            .retain(&third, third.tokens.clone(), 3, 1, MethodCheckpoint::Plain)
            .unwrap());
        assert_eq!(retention.len(), 2);
        assert_eq!(retention.max_entries(), 2);
        assert!(retention.lookup(&first).unwrap().is_none());
        assert!(retention.lookup(&second).unwrap().is_some());
        assert!(retention.lookup(&third).unwrap().is_some());
    }

    #[test]
    fn submitted_entries_can_block_count_capacity_without_overcommit() {
        let mut retention = Retention::new(capacity(1_000, 1));
        let first = text_request(64, "first");
        let second = text_request(64, "second");
        assert!(retention
            .retain(&first, first.tokens.clone(), 1, 1, MethodCheckpoint::Plain)
            .unwrap());
        let hit = retention.lookup(&first).unwrap().unwrap();
        retention.begin_submitted(hit.id()).unwrap();
        assert!(!retention
            .retain(
                &second,
                second.tokens.clone(),
                2,
                1,
                MethodCheckpoint::Plain
            )
            .unwrap());
        assert_eq!(retention.len(), 1);
        assert!(retention.lookup(&second).unwrap().is_none());
    }
}
