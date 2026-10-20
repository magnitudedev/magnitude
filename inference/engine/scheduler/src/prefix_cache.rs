//! The prefix cache: resumable states keyed by the input path they consumed.
//!
//! Every entry is one resume state at the end of a path: the tokens up to its
//! position and the media identities conditioned on them. The cache alone
//! decides whether a request may resume from an entry; a [`PrefixHit`] exists
//! only when the entry's path is a prefix of the request's path at an exact
//! boundary, so nothing downstream re-checks it. Entries on one path share
//! their physical prefix (the state store references rows, not copies).
//!
//! One cache belongs to one loaded model, so artifact, tokenizer and state
//! codec are fixed for every path it holds.
//!
//! The cache prices no entry itself: it bounds the entry count, holds entries
//! only while the state every holder occupies stays within the state the
//! memory plan charges its workload (measured by the owner, see
//! [`PrefixCache::release_beyond_workload`]), and supplies victim identity
//! (least recently used, never an entry in submitted use). What an eviction
//! releases is Seismic's charge, observed by the memory heap after the
//! entry's state drops.
use magnitude_executor::{CONVERSATION_CHECKPOINTS, ResourcePlan};
use magnitude_family_contracts::{InputLayout, TokenId};
use magnitude_generation::{Generation, MethodCheckpoint, MethodChoice};
use serde::{Deserialize, Serialize};

pub const MIN_PREFIX_HIT: usize = 64;

/// The fewest rows a planned branch point must save over a request's resume
/// position. A branch point costs a split prefill chunk and a retained
/// recurrent bank; below this the recomputed prefill is cheaper.
pub const MIN_BRANCH_GAIN: usize = 128;

/// Whether a request takes part in the prefix cache (`cache_prompt`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrefixRetention {
    /// The request neither resumes from nor contributes to the cache.
    Transient,
    /// The request resumes from the cache and contributes its caching
    /// points. `cache_points` are prompt positions, each an exact boundary
    /// below the prompt's end, where the host expects later requests to
    /// diverge (the boundary opening the last message): the request
    /// retains a state at each one, so a later request diverging there
    /// resumes from it instead of recomputing the shared prefix.
    Retain { cache_points: Vec<usize> },
}

/// The most entries the cache holds, and the state the memory plan charges
/// its workload (one conversation at the fit depth with the checkpoints it
/// retains): entries beyond the latest conversation's
/// [`CONVERSATION_CHECKPOINTS`] are held only while the state every holder
/// occupies stays within it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixCacheCapacity {
    pub max_entries: usize,
    pub workload_state_bytes: u64,
}

impl PrefixCacheCapacity {
    pub const fn disabled() -> Self {
        Self {
            max_entries: 0,
            workload_state_bytes: 0,
        }
    }

    pub fn from_resource_plan(plan: &ResourcePlan) -> Self {
        let capacity = plan.capacity();
        Self {
            max_entries: capacity.retention_entries,
            workload_state_bytes: capacity.workload_state_bytes,
        }
    }
}

/// A request's input path: its prompt, then its accepted output, with the
/// media identities its layout conditions on, and the method it generates
/// under. Derived from the generation that owns them, never stored.
#[derive(Clone, Copy)]
pub struct PrefixPath<'a> {
    prompt: &'a [TokenId],
    generated: &'a [TokenId],
    layout: &'a InputLayout,
    method: MethodChoice,
}

impl<'a> PrefixPath<'a> {
    pub fn new(
        prompt: &'a [TokenId],
        generated: &'a [TokenId],
        layout: &'a InputLayout,
        method: MethodChoice,
    ) -> Self {
        Self {
            prompt,
            generated,
            layout,
            method,
        }
    }

    pub fn of(generation: &'a Generation) -> Self {
        Self::new(
            generation.prompt(),
            generation.generated(),
            generation.layout(),
            generation.method_choice(),
        )
    }

    pub fn len(&self) -> usize {
        self.prompt.len() + self.generated.len()
    }

    fn tokens(&self) -> impl Iterator<Item = &'a TokenId> {
        self.prompt.iter().chain(self.generated)
    }

    /// Whether `position` is a legal resume point: at most the path length
    /// and not inside an indivisible media span.
    pub fn exact_boundary(&self, position: usize) -> bool {
        position <= self.len() && self.layout.boundary(position)
    }

    /// The media identities conditioned before an exact boundary.
    fn conditioning_at(&self, position: usize) -> impl Iterator<Item = &'a str> {
        self.layout
            .spans()
            .iter()
            .take_while(move |span| span.start < position)
            .map(|span| span.identity.as_str())
    }

    /// The deepest exact boundary of this path up to which a held path (its
    /// tokens and the media identities conditioned on them) is the same
    /// input. Image content is part of the identity: equal placeholder tokens
    /// with different media diverge at the image.
    pub fn shared_boundary(&self, tokens: &[TokenId], conditioning: &[String]) -> usize {
        self.shared(tokens.iter(), conditioning.iter().map(String::as_str))
    }

    /// [`Self::shared_boundary`] with a live request's prompt, the part of
    /// its path it can cache at a branch point.
    pub fn shared_with_prompt_of(&self, live: PrefixPath) -> usize {
        self.shared(
            live.prompt.iter(),
            live.layout.spans().iter().map(|span| span.identity.as_str()),
        )
    }

    fn shared<'b>(
        &self,
        tokens: impl Iterator<Item = &'b TokenId>,
        conditioning: impl Iterator<Item = &'b str> + Clone,
    ) -> usize {
        let common = self
            .tokens()
            .zip(tokens)
            .take_while(|(left, right)| left == right)
            .count();
        (0..=common)
            .rev()
            .find(|&position| {
                self.exact_boundary(position) && {
                    let mut held = conditioning.clone();
                    self.conditioning_at(position)
                        .all(|own| held.next().is_some_and(|held| held == own))
                }
            })
            .unwrap_or(0)
    }
}

/// Which caching point an entry was retained at (scheduler policy).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CachePoint {
    /// A planned branch point or declared cache point inside the prompt.
    Branch,
    /// The prompt one row before its end.
    PromptBoundary,
    /// A finished request's whole path.
    Terminal,
}

/// A match produced by [`PrefixCache::lookup`]: the entry's path is a prefix
/// of the queried path, ending at an exact boundary of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixHit {
    id: u64,
    position: usize,
}

impl PrefixHit {
    pub const fn id(self) -> u64 {
        self.id
    }

    pub const fn position(self) -> usize {
        self.position
    }
}

pub struct CachedPrefix<S> {
    id: u64,
    point: CachePoint,
    state: S,
    method: MethodCheckpoint,
    tokens: Vec<TokenId>,
    conditioning: Vec<String>,
    last_use: u64,
    submitted: usize,
}

impl<S> CachedPrefix<S> {
    pub const fn state(&self) -> &S {
        &self.state
    }

    pub const fn method(&self) -> &MethodCheckpoint {
        &self.method
    }

    pub fn position(&self) -> usize {
        self.tokens.len()
    }
}

/// Thread-confined prefix cache. States are owned by the cache; a hit only
/// borrows the entry while the executor forks its numerical state and the
/// generation restores the owned method checkpoint.
pub struct PrefixCache<S> {
    capacity: PrefixCacheCapacity,
    clock: u64,
    next_id: u64,
    entries: Vec<CachedPrefix<S>>,
}

impl<S> PrefixCache<S> {
    pub const fn new(capacity: PrefixCacheCapacity) -> Self {
        Self {
            capacity,
            clock: 0,
            next_id: 1,
            entries: Vec::new(),
        }
    }

    pub const fn max_entries(&self) -> usize {
        self.capacity.max_entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Every cached state, for a census of the holdings they pin.
    pub fn states(&self) -> impl Iterator<Item = &S> {
        self.entries.iter().map(|entry| &entry.state)
    }

    pub fn entry(&self, hit: PrefixHit) -> Result<&CachedPrefix<S>, String> {
        self.entries
            .iter()
            .find(|entry| entry.id == hit.id)
            .ok_or_else(|| "prefix hit is no longer cached".to_owned())
    }

    /// The deepest entry whose path is a prefix of `path`, retained under
    /// the method `path` generates under (a request drafting without media
    /// and one generating plainly beside it never share a drafter history).
    /// Entries in
    /// submitted use are hits too: every hit forks shared claims. Only
    /// entries below position `bound` qualify: a resumed request must still
    /// compute the row it samples from.
    pub fn lookup(&mut self, path: PrefixPath, bound: usize) -> Result<Option<PrefixHit>, String> {
        let Some(index) = self.best(path, bound) else {
            return Ok(None);
        };
        let last_use = self.tick()?;
        let entry = &mut self.entries[index];
        entry.last_use = last_use;
        Ok(Some(PrefixHit {
            id: entry.id,
            position: entry.position(),
        }))
    }

    /// The position [`Self::lookup`] would resume `path` at, or zero,
    /// without counting as a use.
    pub fn deepest(&self, path: PrefixPath, bound: usize) -> usize {
        self.best(path, bound)
            .map_or(0, |index| self.entries[index].position())
    }

    fn best(&self, path: PrefixPath, bound: usize) -> Option<usize> {
        let mut best: Option<(usize, usize, u64)> = None;
        for (index, entry) in self.entries.iter().enumerate() {
            let position = entry.position();
            if position < MIN_PREFIX_HIT
                || position >= bound
                || !entry.method.resumes(path.method)
                || path.shared_boundary(&entry.tokens, &entry.conditioning) != position
            {
                continue;
            }
            if best.is_none_or(|(_, best_position, last_use)| {
                position > best_position || (position == best_position && entry.last_use > last_use)
            }) {
                best = Some((index, position, entry.last_use));
            }
        }
        best.map(|(index, _, _)| index)
    }

    /// Whether an entry ends exactly at `position` on `path`, without
    /// counting as a use.
    pub fn holds(&self, path: PrefixPath, position: usize) -> bool {
        self.best(path, position + 1)
            .is_some_and(|index| self.entries[index].position() == position)
    }

    /// The deepest boundary to which `path` shares any cached path, whether
    /// or not an entry ends there.
    pub fn shared_boundary(&self, path: PrefixPath) -> usize {
        self.entries
            .iter()
            .map(|entry| path.shared_boundary(&entry.tokens, &entry.conditioning))
            .max()
            .unwrap_or(0)
    }

    /// Cache `state` as the prefix of `path` ending at `position`, an exact
    /// boundary. An identical path already cached is refreshed instead.
    /// Least-recently-used entries without submitted uses are evicted until
    /// the entry count fits its bound; returns whether the state was cached.
    pub fn retain(
        &mut self,
        path: PrefixPath,
        position: usize,
        point: CachePoint,
        state: S,
        method: MethodCheckpoint,
    ) -> Result<bool, String> {
        if position == 0 || !path.exact_boundary(position) {
            return Err("a cached prefix must end at an exact boundary of its path".into());
        }
        if self.capacity.max_entries == 0 {
            return Ok(false);
        }
        let tokens = path.tokens().take(position).copied().collect::<Vec<_>>();
        let conditioning = path
            .conditioning_at(position)
            .map(str::to_owned)
            .collect::<Vec<_>>();
        if let Some(existing) = self
            .entries
            .iter()
            .position(|entry| {
                entry.tokens == tokens
                    && entry.conditioning == conditioning
                    && std::mem::discriminant(&entry.method) == std::mem::discriminant(&method)
            })
        {
            let last_use = self.tick()?;
            self.entries[existing].last_use = last_use;
            return Ok(false);
        }
        let last_use = self.tick()?;
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or("prefix cache entry identity exhausted")?;
        self.entries.push(CachedPrefix {
            id,
            point,
            state,
            method,
            tokens,
            conditioning,
            last_use,
            submitted: 0,
        });
        while self.entries.len() > self.capacity.max_entries {
            if !self.evict_one_except(Some(id)) {
                self.entries.retain(|entry| entry.id != id);
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub fn begin_submitted(&mut self, id: u64) -> Result<(), String> {
        let entry = self
            .entries
            .iter_mut()
            .find(|entry| entry.id == id)
            .ok_or("prefix cache entry is no longer cached")?;
        entry.submitted = entry
            .submitted
            .checked_add(1)
            .ok_or("prefix cache submitted-use count exhausted")?;
        Ok(())
    }

    pub fn begin_submitted_if_cached(&mut self, id: u64) -> Result<bool, String> {
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
            .ok_or("prefix cache entry is no longer cached")?;
        entry.submitted = entry
            .submitted
            .checked_sub(1)
            .ok_or("prefix cache entry has no outstanding submitted use")?;
        Ok(())
    }

    /// Capacity-ladder rung zero: drop the least-recently-used entry without
    /// submitted uses. Returns whether an entry was eligible; the heap
    /// observes what the drop released.
    pub fn evict_one(&mut self) -> bool {
        self.evict_one_except(None)
    }

    /// Hold cached prefixes within the state the memory plan charges: while
    /// `occupied` (the state every holder occupies, these entries included)
    /// exceeds it, drop the least-recently-used entry without submitted uses,
    /// sparing the [`CONVERSATION_CHECKPOINTS`] the plan charges the latest
    /// conversation (see [`Self::conversation_checkpoints`]). Returns how
    /// many entries were dropped.
    pub fn release_beyond_workload<E>(
        &mut self,
        mut occupied: impl FnMut(&Self) -> Result<u64, E>,
    ) -> Result<usize, E> {
        let mut released = 0;
        while occupied(self)? > self.capacity.workload_state_bytes {
            let spared = self.conversation_checkpoints();
            let candidate = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, entry)| entry.submitted == 0 && !spared.contains(&entry.id))
                .min_by_key(|(_, entry)| entry.last_use)
                .map(|(index, _)| index);
            let Some(index) = candidate else {
                break;
            };
            self.entries.remove(index);
            released += 1;
        }
        Ok(released)
    }

    /// The checkpoints the plan charges the most recently cached
    /// conversation: its latest finished turn, which its next turn extends;
    /// its latest prompt boundary, which a repeat of the prompt resumes from;
    /// and the deepest branch point on the latest entry's path, where its
    /// last turn began: a template that renders earlier replies anew once a
    /// new user message arrives makes the next turn diverge there.
    fn conversation_checkpoints(&self) -> Vec<u64> {
        let latest = |point: CachePoint| {
            self.entries
                .iter()
                .filter(|entry| entry.point == point)
                .max_by_key(|entry| entry.id)
        };
        let terminal = latest(CachePoint::Terminal);
        let boundary = latest(CachePoint::PromptBoundary);
        let path = self.entries.iter().max_by_key(|entry| entry.id);
        let branch = path.and_then(|path| {
            self.entries
                .iter()
                .filter(|entry| {
                    entry.point == CachePoint::Branch
                        && path.tokens.starts_with(&entry.tokens)
                        && path.conditioning.starts_with(&entry.conditioning)
                })
                .max_by_key(|entry| (entry.position(), entry.id))
        });
        [terminal, boundary, branch]
            .into_iter()
            .flatten()
            .map(|entry| entry.id)
            .collect()
    }

    /// Drop every entry without submitted uses.
    pub fn evict_all(&mut self) {
        self.entries.retain(|entry| entry.submitted != 0);
    }

    fn evict_one_except(&mut self, keep: Option<u64>) -> bool {
        let candidate = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.submitted == 0 && Some(entry.id) != keep)
            .min_by_key(|(_, entry)| entry.last_use)
            .map(|(index, _)| index);
        let Some(index) = candidate else {
            return false;
        };
        self.entries.remove(index);
        true
    }

    fn tick(&mut self) -> Result<u64, String> {
        self.clock = self
            .clock
            .checked_add(1)
            .ok_or("prefix cache LRU clock exhausted")?;
        Ok(self.clock)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_family_contracts::{BoundaryRule, InputSpan};
    use std::collections::{BTreeMap, BTreeSet};

    const fn capacity(max_entries: usize) -> PrefixCacheCapacity {
        PrefixCacheCapacity {
            max_entries,
            workload_state_bytes: u64::MAX,
        }
    }

    thread_local! {
        /// References on each abstract row by every live `Rows` value.
        static REFERENCES: std::cell::RefCell<BTreeMap<u32, usize>> =
            const { std::cell::RefCell::new(BTreeMap::new()) };
    }

    /// A test state referencing abstract rows, like a resume state
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

    /// Whether a live state still references `row`: an evicted entry's state
    /// is dropped, which is the release the heap observes.
    fn referenced(row: u32) -> bool {
        REFERENCES.with_borrow(|references| references.get(&row).is_some_and(|count| *count > 0))
    }

    /// An owned request path: prompt, accepted output and layout.
    struct Owned {
        prompt: Vec<TokenId>,
        generated: Vec<TokenId>,
        layout: InputLayout,
    }

    impl Owned {
        fn path(&self) -> PrefixPath<'_> {
            self.under(MethodChoice::Plain)
        }

        fn under(&self, method: MethodChoice) -> PrefixPath<'_> {
            PrefixPath::new(&self.prompt, &self.generated, &self.layout, method)
        }
    }

    fn tokens(tokens: Vec<u32>) -> Owned {
        let count = tokens.len();
        Owned {
            prompt: tokens.into_iter().map(TokenId).collect(),
            generated: Vec::new(),
            layout: InputLayout::new(count, vec![]).unwrap(),
        }
    }

    /// A text prompt of `count` tokens; distinct `name`s are distinct paths.
    fn text(count: usize, name: &str) -> Owned {
        let base = u32::from(name.as_bytes()[0]) * 10_000;
        tokens((base..base + count as u32).collect())
    }

    /// `system` shared tokens, then a question distinguished by `question`.
    fn chat(system: u32, question: u32, length: u32) -> Owned {
        tokens(
            (0..system)
                .chain((system..length).map(|token| token + 1000 * question))
                .collect(),
        )
    }

    fn retain(cache: &mut PrefixCache<Rows>, owned: &Owned, position: usize, state: Rows) -> bool {
        cache
            .retain(owned.path(), position, CachePoint::Branch, state, MethodCheckpoint::Plain)
            .unwrap()
    }

    fn lookup_any(cache: &mut PrefixCache<Rows>, owned: &Owned) -> Option<PrefixHit> {
        cache.lookup(owned.path(), usize::MAX).unwrap()
    }

    #[test]
    fn path_conditioning_follows_exact_boundaries() {
        let owned = Owned {
            prompt: (0..10).map(TokenId).collect(),
            generated: vec![TokenId(99)],
            layout: InputLayout::new(
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
        };
        let path = owned.path();
        assert_eq!(path.len(), 11);
        assert_eq!(path.conditioning_at(2).count(), 0);
        assert_eq!(path.conditioning_at(3).collect::<Vec<_>>(), ["first"]);
        assert_eq!(path.conditioning_at(11).collect::<Vec<_>>(), ["first", "second"]);
        assert!(!path.exact_boundary(8));
        assert!(path.exact_boundary(11));
        assert!(!path.exact_boundary(12));
    }

    /// A request generating plainly beside a drafting load (one with media
    /// its drafter cannot enter) retains a state without drafter history; a
    /// drafting request never resumes it, nor refreshes it away.
    #[test]
    fn hits_follow_the_method_the_entry_was_retained_under() {
        let mut cache = PrefixCache::new(capacity(8));
        let target = text(96, "artifact");
        assert!(retain(&mut cache, &target, 80, rows(0..80)));
        let drafting = MethodChoice::Mtp { proposals: 2 };
        assert!(cache.lookup(target.under(drafting), usize::MAX).unwrap().is_none());
        assert_eq!(cache.deepest(target.under(drafting), usize::MAX), 0);
        assert!(!cache.holds(target.under(drafting), 80));
        assert_eq!(lookup_any(&mut cache, &target).unwrap().position(), 80);
        assert!(cache
            .lookup(target.under(MethodChoice::DFlash { proposals: 2 }), usize::MAX)
            .unwrap()
            .is_none());
    }

    #[test]
    fn index_selects_the_deepest_prefix_entry_and_enforces_minimum_hit() {
        let mut cache = PrefixCache::new(capacity(8));
        let target = text(96, "artifact");
        assert!(retain(&mut cache, &target, 63, rows(0..63)));
        assert!(retain(&mut cache, &target, 64, rows(0..64)));
        assert!(retain(&mut cache, &target, 80, rows(0..80)));

        let hit = lookup_any(&mut cache, &target).unwrap();
        assert_eq!(hit.position(), 80);
        assert_eq!(cache.entry(hit).unwrap().state(), &rows(0..80));
        assert!(lookup_any(&mut cache, &text(63, "artifact")).is_none());
        assert!(lookup_any(&mut cache, &text(96, "other")).is_none());
        // A resumed request must compute the row it samples from: only
        // entries below the bound are hits.
        let below = |cache: &mut PrefixCache<_>, bound| {
            cache
                .lookup(target.path(), bound)
                .unwrap()
                .map(|hit| hit.position())
        };
        assert_eq!(below(&mut cache, 81), Some(80));
        assert_eq!(below(&mut cache, 80), Some(64));
        assert_eq!(below(&mut cache, 64), None);
    }

    /// A finished turn's prefix covers its prompt and its reply; the next
    /// turn's prompt extends that path and resumes at its end.
    #[test]
    fn a_terminal_prefix_seeds_the_next_turn() {
        let mut turn = text(64, "artifact");
        turn.generated = (100_000..100_008).map(TokenId).collect();
        let mut cache = PrefixCache::new(capacity(8));
        assert!(retain(&mut cache, &turn, 72, rows(0..72)));

        let mut next = turn.prompt.clone();
        next.extend(&turn.generated);
        next.extend((200_000..200_010).map(TokenId));
        let next = tokens(next.into_iter().map(|token| token.0).collect());
        let hit = lookup_any(&mut cache, &next).unwrap();
        assert_eq!(hit.position(), 72);
        assert_eq!(cache.entry(hit).unwrap().state(), &rows(0..72));
    }

    /// Shared system prompt, divergent questions: a state at the end of the
    /// system prompt serves every question, and the cache reports how far a
    /// path shares a cached path that has no entry at the divergence.
    #[test]
    fn divergent_requests_share_the_deepest_common_prefix() {
        let mut cache = PrefixCache::new(capacity(8));
        let first = chat(200, 1, 260);
        assert!(retain(&mut cache, &first, 260, rows(0..260)));
        let second = chat(200, 2, 250);
        assert!(lookup_any(&mut cache, &second).is_none());
        assert_eq!(cache.shared_boundary(second.path()), 200);
        // A branch state at the divergence serves the next question.
        assert!(retain(&mut cache, &second, 200, rows(0..200)));
        let third = chat(200, 3, 300);
        let hit = lookup_any(&mut cache, &third).unwrap();
        assert_eq!(hit.position(), 200);
        assert_eq!(cache.len(), 2);
    }

    /// A state retained where the last message begins (a declared cache
    /// point) serves every later question, whatever it shares with earlier
    /// ones beyond that point; `holds` reports it without counting a use.
    #[test]
    fn a_state_at_the_last_message_serves_every_changed_message() {
        let mut cache = PrefixCache::new(capacity(8));
        let first = chat(200, 1, 260);
        assert!(!cache.holds(first.path(), 200));
        assert!(retain(&mut cache, &first, 200, rows(0..200)));
        assert!(retain(&mut cache, &first, 260, rows(0..260)));
        assert!(cache.holds(first.path(), 200));
        assert!(cache.holds(first.path(), 260));
        assert!(!cache.holds(first.path(), 230));
        for question in 2..5 {
            let next = chat(200, question, 240 + question);
            assert!(cache.holds(next.path(), 200));
            assert!(!cache.holds(next.path(), 260));
            assert_eq!(lookup_any(&mut cache, &next).unwrap().position(), 200);
        }
        assert!(!cache.holds(text(260, "other").path(), 200));
    }

    #[test]
    fn identical_paths_are_cached_once() {
        let mut cache = PrefixCache::new(capacity(8));
        let owned = text(100, "artifact");
        assert!(retain(&mut cache, &owned, 100, rows(0..100)));
        assert!(!retain(&mut cache, &owned, 100, rows(500..600)));
        assert_eq!(cache.len(), 1);
        // The duplicate's state is dropped; the first one is kept.
        assert!(referenced(0));
        assert!(!referenced(500));
    }

    #[test]
    fn a_prefix_must_end_at_an_exact_boundary() {
        let mut cache = PrefixCache::new(capacity(8));
        let owned = text(100, "artifact");
        assert!(cache
            .retain(owned.path(), 0, CachePoint::Branch, rows(0..0), MethodCheckpoint::Plain)
            .is_err());
        assert!(cache
            .retain(owned.path(), 101, CachePoint::Branch, rows(0..101), MethodCheckpoint::Plain)
            .is_err());
    }

    #[test]
    fn index_rejects_conditioning_identity_mismatch() {
        let conditioned = |identity: &str| Owned {
            prompt: (0..80).map(TokenId).collect(),
            generated: Vec::new(),
            layout: InputLayout::new(
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
        };
        let source = conditioned("source");
        let mut cache = PrefixCache::new(capacity(8));
        assert!(retain(&mut cache, &source, 80, rows(0..80)));

        let changed = conditioned("changed");
        assert!(lookup_any(&mut cache, &changed).is_none());
        // The same tokens with another image diverge before the image.
        assert_eq!(cache.shared_boundary(changed.path()), 2);
        assert_eq!(cache.shared_boundary(conditioned("source").path()), 80);
    }

    #[test]
    fn entry_limit_evicts_oldest_use_and_never_evicts_submitted_entries() {
        let mut cache = PrefixCache::new(capacity(2));
        let first = text(64, "first");
        let second = text(64, "second");
        let third = text(64, "third");
        assert!(retain(&mut cache, &first, 64, rows(0..64)));
        assert!(retain(&mut cache, &second, 64, rows(100..164)));
        let first_hit = lookup_any(&mut cache, &first).unwrap();
        cache.begin_submitted(first_hit.id()).unwrap();
        // Concurrent requests share an entry in submitted use.
        assert_eq!(lookup_any(&mut cache, &first), Some(first_hit));
        assert!(retain(&mut cache, &third, 64, rows(200..264)));
        assert_eq!(cache.len(), 2);
        assert!(!referenced(100));
        assert!(lookup_any(&mut cache, &second).is_none());
        assert!(lookup_any(&mut cache, &third).is_some());
        cache.end_submitted(first_hit.id()).unwrap();
        assert!(lookup_any(&mut cache, &first).is_some());
    }

    #[test]
    fn eviction_drops_least_recent_states_and_keeps_submitted_ones() {
        let mut cache = PrefixCache::new(capacity(8));
        let owned = text(100, "artifact");
        for position in [64, 80, 100] {
            assert!(retain(
                &mut cache,
                &owned,
                position,
                rows(1000 + position as u32..1001 + position as u32)
            ));
        }
        assert_eq!(cache.len(), 3);
        assert!(cache.evict_one());
        assert_eq!(cache.len(), 2);
        assert!(!referenced(1064));
        assert!(referenced(1080) && referenced(1100));
        let hit = lookup_any(&mut cache, &owned).unwrap();
        assert_eq!(hit.position(), 100);
        cache.begin_submitted(hit.id()).unwrap();
        cache.evict_all();
        assert_eq!(cache.len(), 1);
        assert!(!referenced(1080) && referenced(1100));
        assert!(!cache.evict_one());
    }

    #[test]
    fn numerical_entry_capacity_evicts() {
        let mut cache = PrefixCache::new(capacity(2));
        let first = text(64, "first");
        let second = text(64, "second");
        let third = text(64, "third");
        assert!(retain(&mut cache, &first, 64, rows(0..1)));
        assert!(retain(&mut cache, &second, 64, rows(1..2)));
        assert!(retain(&mut cache, &third, 64, rows(2..3)));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.max_entries(), 2);
        assert!(lookup_any(&mut cache, &first).is_none());
        assert!(lookup_any(&mut cache, &second).is_some());
        assert!(lookup_any(&mut cache, &third).is_some());
    }

    /// An agent conversation of growing user turns, each a tool call and an
    /// answer, under a template that renders the previous user turn's
    /// replies anew once a new user message arrives (as Qwen3.5's does): each
    /// request retains its prompt boundary and finished turn, and the first
    /// of a user turn also the declared point opening its user message.
    /// Cached prefixes never take the stores past the workload the plan
    /// charges, and every request still resumes where it would without a
    /// bound: a tool result at the previous reply, a new user turn at the
    /// point opening the previous user message, even far beyond the depth.
    #[test]
    fn a_growing_conversation_stays_within_the_charged_workload() {
        // Rows are history rows named by the token they hold; a bank is one
        // id from `BANK` on, weighing `BANK_ROWS` rows.
        const BANK: u32 = 1_000_000;
        const BANK_ROWS: u64 = 50;
        const DEPTH: u64 = 2000;
        const THINK: usize = 4;
        let workload = DEPTH + 6 * BANK_ROWS;
        let occupied = |_: &PrefixCache<Rows>| -> Result<u64, String> {
            Ok(REFERENCES.with_borrow(|references| {
                references
                    .iter()
                    .filter(|(_, count)| **count > 0)
                    .map(|(&row, _)| if row >= BANK { BANK_ROWS } else { 1 })
                    .sum()
            }))
        };
        let mut cache = PrefixCache::new(PrefixCacheCapacity {
            max_entries: 1024,
            workload_state_bytes: workload,
        });
        let next_token = std::cell::Cell::new(1u32);
        let fresh = |count: usize| {
            let first = next_token.get();
            next_token.set(first + count as u32);
            (first..first + count as u32).collect::<Vec<_>>()
        };
        let banks = std::cell::Cell::new(BANK);
        let cache_at = |cache: &mut PrefixCache<Rows>, path: &[u32], position: usize, point| {
            banks.set(banks.get() + 1);
            let state = Rows::new(path[..position].iter().copied().chain([banks.get()]).collect());
            let owned = tokens(path.to_vec());
            assert!(cache
                .retain(owned.path(), position, point, state, MethodCheckpoint::Plain)
                .unwrap());
            cache.release_beyond_workload(occupied).unwrap();
        };
        // Each request's prompt, what it resumed at, and its reply.
        let request = |cache: &mut PrefixCache<Rows>, prompt: &[u32], declared: Option<usize>| {
            let resumed = lookup_any(cache, &tokens(prompt.to_vec())).map(|hit| hit.position());
            if let Some(point) = declared {
                cache_at(cache, prompt, point, CachePoint::Branch);
            }
            cache_at(cache, prompt, prompt.len() - 1, CachePoint::PromptBoundary);
            let mut finished = prompt.to_vec();
            finished.extend(fresh(20));
            cache_at(cache, &finished, finished.len(), CachePoint::Terminal);
            (resumed, finished)
        };
        let mut history = fresh(100);
        // Where the previous user turn's message and replies begin.
        let mut previous: Option<(usize, usize)> = None;
        for turn in 0..30 {
            if let Some((_, replies)) = previous {
                // The previous replies render anew from their first one on.
                let rendered = history.len() - replies;
                history.truncate(replies);
                history.extend(fresh(rendered));
            }
            let user = history.len();
            history.extend(fresh(150));
            // The generation prompt ends with an empty reasoning block, which
            // earlier replies render without.
            history.extend(fresh(THINK));
            let (resumed, finished) = request(&mut cache, &history, Some(user));
            assert_eq!(
                resumed,
                previous.map(|(user, _)| user),
                "turn {turn} resumes where the previous user message opens"
            );
            let replies = user + 150;
            let mut prompt = finished.clone();
            prompt.extend(fresh(300));
            let (resumed, finished) = request(&mut cache, &prompt, None);
            assert_eq!(resumed, Some(prompt.len() - 300), "turn {turn}'s tool result resumes at the call");
            let held = occupied(&cache).unwrap();
            assert!(
                held <= workload || cache.len() <= CONVERSATION_CHECKPOINTS,
                "turn {turn}: {} entries occupy {held} of {workload}",
                cache.len()
            );
            history = finished;
            previous = Some((user, replies));
        }
        assert!(history.len() as u64 > 4 * DEPTH);
        assert_eq!(cache.len(), CONVERSATION_CHECKPOINTS);
    }

    #[test]
    fn submitted_entries_can_block_count_capacity_without_overcommit() {
        let mut cache = PrefixCache::new(capacity(1));
        let first = text(64, "first");
        let second = text(64, "second");
        assert!(retain(&mut cache, &first, 64, rows(0..1)));
        let hit = lookup_any(&mut cache, &first).unwrap();
        cache.begin_submitted(hit.id()).unwrap();
        assert!(!retain(&mut cache, &second, 64, rows(1..2)));
        assert_eq!(cache.len(), 1);
        assert!(lookup_any(&mut cache, &second).is_none());
    }
}
