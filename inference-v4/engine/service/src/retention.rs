//! The token-prefix index of retained numerical checkpoints.
//!
//! Every entry is one checkpoint on a token path: the interpreted tokens up to
//! its position and the media identities conditioned before it. Entries on
//! one path share their physical prefix (the state store references rows,
//! not copies), so the retention budget prices the whole retained set by the
//! bytes it alone holds: a shared prefix is charged once, and not at all
//! while a live request still shares it.
use magnitude_artifacts::{InputLayout, PackageIdentity, TokenId};
use magnitude_generation::MethodCheckpoint;
use magnitude_model_contracts::{PreparedModelInput, TokenPlan};
use magnitude_model_executor::ResourcePlan;
use magnitude_model_state::CodecIdentity;
use std::sync::Arc;

pub const MIN_RETENTION_HIT: usize = 64;

/// The fewest rows a planned branch point must save over a request's resume
/// position. A branch point costs a split prefill chunk and a retained
/// recurrent bank; below this the recomputed prefill is cheaper.
pub const MIN_BRANCH_GAIN: usize = 128;

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

    /// The deepest exact input boundary of this request up to which a held
    /// path (its tokens and the media identities conditioned on them) is the
    /// same input. Image content is part of the identity: equal placeholder
    /// tokens with different media diverge at the image.
    pub fn shared_boundary(&self, tokens: &[TokenId], conditioning: &[String]) -> usize {
        let common = self
            .tokens
            .iter()
            .zip(tokens)
            .take_while(|(left, right)| left == right)
            .count();
        (0..=common)
            .rev()
            .find(|&position| {
                self.conditioning_at(position).is_ok_and(|own| {
                    own.len() <= conditioning.len() && own == &conditioning[..own.len()]
                })
            })
            .unwrap_or(0)
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
}

/// Thread-confined token-prefix index. Checkpoints are owned by the index; a
/// hit only borrows the entry while the executor forks its numerical state and
/// the fresh generation restores the owned method checkpoint.
///
/// `meter` arguments price a set of checkpoints: the physical bytes released
/// if exactly that set were dropped.
pub struct Retention<C> {
    capacity: RetentionCapacity,
    charged: u64,
    clock: u64,
    next_id: u64,
    entries: Vec<RetainedEntry<C>>,
}

impl<C> Retention<C> {
    pub const fn new(capacity: RetentionCapacity) -> Self {
        Self {
            capacity,
            charged: 0,
            clock: 0,
            next_id: 1,
            entries: Vec::new(),
        }
    }

    pub const fn budget_bytes(&self) -> u64 {
        self.capacity.max_bytes
    }

    pub const fn max_entries(&self) -> usize {
        self.capacity.max_entries
    }

    /// Bytes the retained set alone held when it last changed.
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

    /// The deepest entry whose path is a prefix of the request. Entries in
    /// submitted use are hits too: every hit forks shared claims.
    /// Only entries below position `before` qualify: a resumed request must
    /// still compute the row it samples from (the last prompt row at
    /// admission, the last accepted row after eviction).
    pub fn lookup(
        &mut self,
        request: &RetentionRequest,
        before: usize,
    ) -> Result<Option<RetentionHit>, String> {
        let mut best = None;
        for (index, entry) in self.entries.iter().enumerate() {
            if entry.key != request.key
                || entry.position < MIN_RETENTION_HIT
                || entry.position >= before
                || request.shared_boundary(&entry.tokens, &entry.conditioning) != entry.position
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

    #[cfg(test)]
    fn lookup_any(&mut self, request: &RetentionRequest) -> Result<Option<RetentionHit>, String> {
        self.lookup(request, usize::MAX)
    }

    /// The deepest boundary to which the request shares any retained path,
    /// whether or not a checkpoint exists there.
    pub fn shared_boundary(&self, request: &RetentionRequest) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.key == request.key)
            .map(|entry| request.shared_boundary(&entry.tokens, &entry.conditioning))
            .max()
            .unwrap_or(0)
    }

    /// Retain `checkpoint` at the end of `tokens`: a prefix of the request's
    /// interpreted input ending at an exact boundary (a branch point or the
    /// prompt end) or its extension by generated tokens. An identical path
    /// already retained is
    /// refreshed instead. Least-recently-used entries without submitted uses
    /// are evicted until the retained set's own bytes fit the budget; returns
    /// whether the checkpoint was retained.
    pub fn retain(
        &mut self,
        request: &RetentionRequest,
        tokens: Vec<TokenId>,
        checkpoint: C,
        method: MethodCheckpoint,
        meter: impl Fn(&[&C]) -> Result<u64, String>,
    ) -> Result<bool, String> {
        let position = tokens.len();
        if position == 0
            || !request.exact_boundary(position)
            || !(tokens.starts_with(&request.tokens) || request.tokens.starts_with(&tokens))
        {
            return Err("retained state is not on its request's interpreted input".into());
        }
        let conditioning = request.conditioning_at(position)?.to_vec();
        if self.capacity.max_bytes == 0 || self.capacity.max_entries == 0 {
            return Ok(false);
        }
        if let Some(existing) = self.entries.iter().position(|entry| {
            entry.key == request.key && entry.tokens == tokens && entry.conditioning == conditioning
        }) {
            let last_use = self.tick()?;
            self.entries[existing].last_use = last_use;
            return Ok(false);
        }
        let last_use = self.tick()?;
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or("retention entry identity exhausted")?;
        self.entries.push(RetainedEntry {
            id,
            key: request.key.clone(),
            checkpoint,
            method,
            tokens,
            conditioning,
            position,
            last_use,
            submitted: 0,
        });
        loop {
            let charged = self.measure(&meter)?;
            if charged <= self.capacity.max_bytes && self.entries.len() <= self.capacity.max_entries
            {
                self.charged = charged;
                return Ok(true);
            }
            if self.evict_one_except(Some(id), &meter)?.is_none() {
                self.entries.retain(|entry| entry.id != id);
                self.charged = self.measure(&meter)?;
                return Ok(false);
            }
        }
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

    /// Capacity-ladder rung zero: release the least-recently-used entry
    /// without submitted uses. Returns the bytes its eviction released, or
    /// `None` when no entry is eligible.
    pub fn evict_one(
        &mut self,
        meter: impl Fn(&[&C]) -> Result<u64, String>,
    ) -> Result<Option<u64>, String> {
        let released = self.evict_one_except(None, &meter)?;
        if released.is_some() {
            self.charged = self.measure(&meter)?;
        }
        Ok(released)
    }

    /// Re-price the retained set after its sharers changed (a live request
    /// that shared a retained prefix ended, so the set alone now holds it) and
    /// evict least-recently-used entries until it fits the budget again.
    pub fn enforce_budget(
        &mut self,
        meter: impl Fn(&[&C]) -> Result<u64, String>,
    ) -> Result<(), String> {
        loop {
            self.charged = self.measure(&meter)?;
            if self.charged <= self.capacity.max_bytes
                || self.evict_one_except(None, &meter)?.is_none()
            {
                return Ok(());
            }
        }
    }

    /// Release every entry without submitted uses.
    pub fn evict_all(&mut self, meter: impl Fn(&[&C]) -> Result<u64, String>) -> Result<u64, String> {
        let mut released = 0u64;
        while let Some(bytes) = self.evict_one_except(None, &meter)? {
            released = released.saturating_add(bytes);
        }
        self.charged = self.measure(&meter)?;
        Ok(released)
    }

    fn evict_one_except(
        &mut self,
        keep: Option<u64>,
        meter: &impl Fn(&[&C]) -> Result<u64, String>,
    ) -> Result<Option<u64>, String> {
        let candidate = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.submitted == 0 && Some(entry.id) != keep)
            .min_by_key(|(_, entry)| entry.last_use)
            .map(|(index, _)| index);
        let Some(index) = candidate else {
            return Ok(None);
        };
        let released = meter(&[&self.entries[index].checkpoint])?
            .checked_add(self.entries[index].method.retained_bytes())
            .ok_or("retention release byte count overflow")?;
        self.entries.remove(index);
        Ok(Some(released))
    }

    fn measure(&self, meter: &impl Fn(&[&C]) -> Result<u64, String>) -> Result<u64, String> {
        let checkpoints = self
            .entries
            .iter()
            .map(|entry| &entry.checkpoint)
            .collect::<Vec<_>>();
        self.entries.iter().try_fold(meter(&checkpoints)?, |total, entry| {
            total
                .checked_add(entry.method.retained_bytes())
                .ok_or_else(|| "retention charge overflow".to_owned())
        })
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
    use std::collections::{BTreeMap, BTreeSet};

    const fn capacity(max_bytes: u64, max_entries: usize) -> RetentionCapacity {
        RetentionCapacity {
            max_bytes,
            max_entries,
        }
    }

    thread_local! {
        /// References on each abstract row by every live `Rows` value.
        static REFERENCES: std::cell::RefCell<BTreeMap<u32, usize>> =
            const { std::cell::RefCell::new(BTreeMap::new()) };
    }

    /// A test checkpoint referencing abstract rows, like a state checkpoint
    /// referencing arena rows.
    #[derive(Debug, PartialEq, Eq)]
    struct Rows(BTreeSet<u32>);

    impl Clone for Rows {
        fn clone(&self) -> Self {
            Rows::new(self.0.clone())
        }
    }

    impl Rows {
        fn new(rows: BTreeSet<u32>) -> Self {
            REFERENCES.with_borrow_mut(|references| {
                for row in &rows {
                    *references.entry(*row).or_default() += 1;
                }
            });
            Self(rows)
        }
    }

    impl Drop for Rows {
        fn drop(&mut self) {
            REFERENCES.with_borrow_mut(|references| {
                for row in &self.0 {
                    *references.get_mut(row).unwrap() -= 1;
                }
            });
        }
    }

    fn rows(range: std::ops::Range<u32>) -> Rows {
        Rows::new(range.collect())
    }

    /// Rows referenced by nothing outside the set, as the state store prices.
    fn meter(set: &[&Rows]) -> Result<u64, String> {
        let mut within = BTreeMap::<u32, usize>::new();
        for rows in set {
            for row in &rows.0 {
                *within.entry(*row).or_default() += 1;
            }
        }
        Ok(REFERENCES.with_borrow(|references| {
            within
                .iter()
                .filter(|(row, count)| references[row] == **count)
                .count() as u64
        }))
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

    fn token_request(tokens: Vec<u32>, artifact: u8) -> RetentionRequest {
        let count = tokens.len();
        RetentionRequest::from_token_plan(
            RetentionKey::new(
                PackageIdentity {
                    target: magnitude_artifacts::ArtifactIdentity([artifact; 32]),
                    projector: None,
                },
                TokenizerIdentity::new("tokenizer").unwrap(),
                CodecIdentity::new("codec").unwrap(),
            ),
            &TokenPlan::new(
                tokens.into_iter().map(TokenId).collect(),
                InputLayout::new(count, vec![]).unwrap(),
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn text_request(count: usize, artifact: &str) -> RetentionRequest {
        token_request(
            (0..count as u32).collect(),
            artifact.as_bytes().first().copied().unwrap_or(0),
        )
    }

    /// `system` shared tokens, then a question distinguished by `question`.
    fn chat(system: u32, question: u32, length: u32) -> RetentionRequest {
        token_request(
            (0..system)
                .chain((system..length).map(|token| token + 1000 * question))
                .collect(),
            b'a',
        )
    }

    fn retain(
        retention: &mut Retention<Rows>,
        request: &RetentionRequest,
        position: usize,
        checkpoint: Rows,
    ) -> bool {
        retention
            .retain(
                request,
                request.tokens[..position].to_vec(),
                checkpoint,
                MethodCheckpoint::Plain,
                meter,
            )
            .unwrap()
    }

    #[test]
    fn index_selects_the_deepest_prefix_entry_and_enforces_minimum_hit() {
        let mut retention = Retention::new(capacity(1_000, 8));
        let target = text_request(96, "artifact");
        assert!(retain(&mut retention, &target, 63, rows(0..63)));
        assert!(retain(&mut retention, &target, 64, rows(0..64)));
        assert!(retain(&mut retention, &target, 80, rows(0..80)));

        let hit = retention.lookup_any(&target).unwrap().unwrap();
        assert_eq!(hit.position(), 80);
        assert_eq!(retention.entry(hit).unwrap().checkpoint(), &rows(0..80));
        assert!(retention
            .lookup_any(&text_request(63, "artifact"))
            .unwrap()
            .is_none());
        assert!(retention
            .lookup_any(&text_request(96, "other"))
            .unwrap()
            .is_none());
        // A resumed request must compute the row it samples from: only
        // entries below the bound are hits.
        let below = |retention: &mut Retention<_>, before| {
            retention.lookup(&target, before).unwrap().map(|hit| hit.position())
        };
        assert_eq!(below(&mut retention, 81), Some(80));
        assert_eq!(below(&mut retention, 80), Some(64));
        assert_eq!(below(&mut retention, 64), None);
    }

    #[test]
    fn terminal_suffix_is_an_exact_boundary_and_can_seed_a_longer_prompt() {
        let source = text_request(64, "artifact");
        let mut terminal = source.tokens.clone();
        terminal.extend((64..72).map(TokenId));
        let mut retention = Retention::new(capacity(1_000, 8));
        assert!(retention
            .retain(&source, terminal, rows(0..72), MethodCheckpoint::Plain, meter)
            .unwrap());

        let target = text_request(80, "artifact");
        let hit = retention.lookup_any(&target).unwrap().unwrap();
        assert_eq!(hit.position(), 72);
        assert_eq!(retention.entry(hit).unwrap().checkpoint(), &rows(0..72));
    }

    /// Shared system prompt, divergent questions: a checkpoint at the end of
    /// the system prompt serves every question, and the index reports how far
    /// a request shares a path that has no checkpoint at the divergence.
    #[test]
    fn divergent_requests_share_the_deepest_common_checkpoint() {
        let mut retention = Retention::new(capacity(10_000, 8));
        let first = chat(200, 1, 260);
        // First turn ended; only its prompt end and terminal are retained.
        assert!(retain(&mut retention, &first, 260, rows(0..260)));
        let second = chat(200, 2, 250);
        assert!(retention.lookup_any(&second).unwrap().is_none());
        assert_eq!(retention.shared_boundary(&second), 200);
        // A branch checkpoint at the divergence serves the next question.
        assert!(retain(&mut retention, &second, 200, rows(0..200)));
        let third = chat(200, 3, 300);
        let hit = retention.lookup_any(&third).unwrap().unwrap();
        assert_eq!(hit.position(), 200);
        // The shared prefix is charged once: 260 path rows + no new rows.
        assert_eq!(retention.charged_bytes(), 260);
    }

    #[test]
    fn identical_paths_are_retained_once() {
        let mut retention = Retention::new(capacity(1_000, 8));
        let request = text_request(100, "artifact");
        assert!(retain(&mut retention, &request, 100, rows(0..100)));
        assert!(!retain(&mut retention, &request, 100, rows(500..600)));
        assert_eq!(retention.len(), 1);
        assert_eq!(retention.charged_bytes(), 100);
    }

    #[test]
    fn index_rejects_conditioning_identity_mismatch() {
        let conditioned = |identity: &str| {
            let plan = TokenPlan::new(
                (0..80).map(TokenId).collect(),
                InputLayout::new(
                    80,
                    vec![InputSpan {
                        start: 2,
                        end: 5,
                        identity: identity.into(),
                        boundaries: BoundaryRule::Indivisible,
                        language_history: false,
                    }],
                )
                .unwrap(),
            )
            .unwrap();
            RetentionRequest::from_token_plan(key(), &plan).unwrap()
        };
        let source = conditioned("source");
        let mut retention = Retention::new(capacity(1_000, 8));
        assert!(retain(&mut retention, &source, 80, rows(0..80)));

        let changed = conditioned("changed");
        assert!(retention.lookup_any(&changed).unwrap().is_none());
        // The same tokens with another image diverge before the image.
        assert_eq!(retention.shared_boundary(&changed), 2);
        assert_eq!(retention.shared_boundary(&conditioned("source")), 80);
    }

    #[test]
    fn budget_evicts_oldest_use_and_never_evicts_submitted_entries() {
        let mut retention = Retention::new(capacity(128, 8));
        let first = text_request(64, "first");
        let second = text_request(64, "second");
        let third = text_request(64, "third");
        assert!(retain(&mut retention, &first, 64, rows(0..64)));
        assert!(retain(&mut retention, &second, 64, rows(100..164)));
        let first_hit = retention.lookup_any(&first).unwrap().unwrap();
        retention.begin_submitted(first_hit.id()).unwrap();
        // Concurrent requests share an entry in submitted use.
        assert_eq!(retention.lookup_any(&first).unwrap(), Some(first_hit));
        assert!(retain(&mut retention, &third, 64, rows(200..264)));
        assert_eq!(retention.charged_bytes(), 128);
        assert!(retention.lookup_any(&second).unwrap().is_none());
        assert!(retention.lookup_any(&third).unwrap().is_some());
        retention.end_submitted(first_hit.id()).unwrap();
        assert!(retention.lookup_any(&first).unwrap().is_some());
    }

    #[test]
    fn a_shared_path_fits_a_budget_its_entries_would_exceed_separately() {
        // Three checkpoints on one 100-row path cost 100 rows, not 300.
        let mut retention = Retention::new(capacity(100, 8));
        let path = text_request(100, "artifact");
        for position in [64, 80, 100] {
            assert!(retain(&mut retention, &path, position, rows(0..position as u32)));
        }
        assert_eq!(retention.len(), 3);
        assert_eq!(retention.charged_bytes(), 100);
        // Evicting an entry whose rows another entry shares releases nothing.
        assert_eq!(retention.evict_one(meter).unwrap(), Some(0));
        assert_eq!(retention.charged_bytes(), 100);
    }

    #[test]
    fn numerical_entry_capacity_evicts_even_when_byte_budget_has_room() {
        let mut retention = Retention::new(capacity(1_000, 2));
        let first = text_request(64, "first");
        let second = text_request(64, "second");
        let third = text_request(64, "third");
        assert!(retain(&mut retention, &first, 64, rows(0..1)));
        assert!(retain(&mut retention, &second, 64, rows(1..2)));
        assert!(retain(&mut retention, &third, 64, rows(2..3)));
        assert_eq!(retention.len(), 2);
        assert_eq!(retention.max_entries(), 2);
        assert!(retention.lookup_any(&first).unwrap().is_none());
        assert!(retention.lookup_any(&second).unwrap().is_some());
        assert!(retention.lookup_any(&third).unwrap().is_some());
    }

    #[test]
    fn submitted_entries_can_block_count_capacity_without_overcommit() {
        let mut retention = Retention::new(capacity(1_000, 1));
        let first = text_request(64, "first");
        let second = text_request(64, "second");
        assert!(retain(&mut retention, &first, 64, rows(0..1)));
        let hit = retention.lookup_any(&first).unwrap().unwrap();
        retention.begin_submitted(hit.id()).unwrap();
        assert!(!retain(&mut retention, &second, 64, rows(1..2)));
        assert_eq!(retention.len(), 1);
        assert!(retention.lookup_any(&second).unwrap().is_none());
    }
}
