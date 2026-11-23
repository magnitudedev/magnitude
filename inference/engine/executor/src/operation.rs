use crate::batching::Demand;
use crate::{ConditioningRef, FeatureRef, ImageRef, LogitsRef};
pub use magnitude_artifacts::TokenId;
use std::{fmt, sync::Arc};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceDomainId(Arc<str>);

macro_rules! identity {
    ($name:ident) => {
        impl $name {
            pub fn new(value: impl Into<Arc<str>>) -> Result<Self, String> {
                let value = value.into();
                if value.is_empty() {
                    return Err(concat!(stringify!($name), " must not be empty").into());
                }
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }
    };
}

identity!(ResourceDomainId);

/// Engine-wide identity of an admitted request: unique in the process, and
/// ordered as requests were admitted across every load
/// ([`Self::admitted`]). A store's ledger, shared by every load in the
/// process, keys each sequence's state by it, so no two loads' sequences
/// share one and "admitted after" is one order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RequestId(pub u64);

impl RequestId {
    /// The identity of a request admitted now: the one way a request's
    /// identity is made, later than every one made before it in the
    /// process.
    pub fn admitted() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        assert!(id != u64::MAX, "invariant violated: request identities exhausted");
        Self(id)
    }
}

/// A contiguous row range within an aggregate feature lease.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FeatureSpan {
    pub features: FeatureRef,
    pub start: usize,
    pub count: usize,
}

/// Feature rows copied to host memory, each `row_bytes` bytes in the
/// activation representation of the domain that produced them. Rows carried
/// across operations live here, so no device lease outlives its round.
#[derive(Clone, PartialEq, Eq)]
pub struct FeatureRows {
    bytes: Arc<[u8]>,
    rows: usize,
}

impl FeatureRows {
    pub fn new(bytes: Arc<[u8]>, rows: usize) -> Result<Self, OperationError> {
        if rows == 0 || bytes.is_empty() || bytes.len() % rows != 0 {
            return Err(OperationError::FeatureRows {
                bytes: bytes.len(),
                rows,
            });
        }
        Ok(Self { bytes, rows })
    }

    /// These rows followed by `other`'s, which must have the same width.
    pub fn concat(&self, other: &Self) -> Result<Self, OperationError> {
        if other.row_bytes() != self.row_bytes() {
            return Err(OperationError::FeatureRows {
                bytes: other.bytes.len(),
                rows: other.rows,
            });
        }
        let mut bytes = Vec::with_capacity(self.bytes.len() + other.bytes.len());
        bytes.extend_from_slice(&self.bytes);
        bytes.extend_from_slice(&other.bytes);
        Self::new(bytes.into(), self.rows + other.rows)
    }

    /// Rows `start..start + count`.
    pub fn slice(&self, start: usize, count: usize) -> Result<Self, OperationError> {
        let row = self.row_bytes();
        if count == 0 || start.checked_add(count).is_none_or(|end| end > self.rows) {
            return Err(OperationError::FeatureRows {
                bytes: self.bytes.len(),
                rows: self.rows,
            });
        }
        Self::new(self.bytes[start * row..(start + count) * row].into(), count)
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub fn row_bytes(&self) -> usize {
        self.bytes.len() / self.rows
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl fmt::Debug for FeatureRows {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FeatureRows")
            .field("rows", &self.rows)
            .field("row_bytes", &self.row_bytes())
            .finish_non_exhaustive()
    }
}

/// Resource-domain seam through which generation copies completed feature
/// rows to host memory.
pub trait FeatureReader {
    fn read(&mut self, span: &FeatureSpan) -> Result<FeatureRows, String>;
}

impl FeatureSpan {
    pub fn new(features: FeatureRef, start: usize, count: usize) -> Result<Self, OperationError> {
        if count == 0 || start.checked_add(count).is_none() {
            return Err(OperationError::FeatureSpan { count, rows: count });
        }
        Ok(Self {
            features,
            start,
            count,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WorkKind {
    Prefill,
    Replay,
    Decode,
    Verify,
}

/// What a target launch's rows are, as policy tells them apart: a prompt's
/// (any of its requests prefills or replays), or generation's (each of its
/// requests decodes or verifies). Independent of the launch's form: a
/// generation launch wider than the decode form runs the grouped form, and a
/// short prompt tail the decode form, and neither is the other's work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaunchWork {
    Prompt,
    Generation,
}

impl LaunchWork {
    /// The work of a launch of requests of `kinds`, at least one: a launch
    /// of no request is the caller's broken invariant.
    pub fn of(kinds: impl IntoIterator<Item = WorkKind>) -> Self {
        let mut kinds = kinds.into_iter().peekable();
        assert!(kinds.peek().is_some(), "a launch runs a request");
        match kinds.any(|kind| matches!(kind, WorkKind::Prefill | WorkKind::Replay)) {
            true => Self::Prompt,
            false => Self::Generation,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Sampling {
    Greedy,
    Categorical,
}

/// Device-side distribution shaping. Identity values leave logits unchanged.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Shaping {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
}

impl Default for Shaping {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        }
    }
}

impl Shaping {
    pub fn validate(self) -> Result<Self, &'static str> {
        let finite = [
            self.temperature,
            self.top_p,
            self.min_p,
            self.repetition_penalty,
            self.presence_penalty,
            self.frequency_penalty,
        ]
        .into_iter()
        .all(f32::is_finite);
        if !(finite
            && self.temperature >= 0.0
            && 0.0 < self.top_p
            && self.top_p <= 1.0
            && (0.0..=1.0).contains(&self.min_p)
            && self.top_k <= (1 << 24)
            && self.repetition_penalty > 0.0)
        {
            return Err("invalid sampling shaping parameters");
        }
        Ok(self)
    }

    pub fn is_identity(self) -> bool {
        self == Self::default()
    }

    pub fn uses_history(self) -> bool {
        self.repetition_penalty != 1.0
            || self.presence_penalty != 0.0
            || self.frequency_penalty != 0.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectSpec {
    pub sampling: Sampling,
    pub seed: u64,
    pub position: usize,
    pub domain: u32,
    pub mask: Option<Arc<[u32]>>,
    pub shaping: Shaping,
    pub history: Option<Arc<[i32]>>,
}

/// A decode pass's rows over one sequence: one exact row, the pending
/// token, which always commits, then its rows ahead in position order, each
/// a guess of the token after the row before it. A pass's rows ahead are
/// either acceptable rows, whose acceptance may extend the commit (a
/// verification's proposals, each checked against the selection of the row
/// before it), or front rows, which only run ahead and are never accepted
/// (a front pass's, which name their tokens' routes early): front row `d`
/// runs on the `d`-th guess of the sequence's chain, the token `d` positions
/// past the pending one, and is exact `d` passes later where every guess up
/// to it stands. Each row sees the K/V and recurrent state of every earlier
/// row of the pass.
///
/// | pass | rows | exact, acceptable, front |
/// |---|---|---|
/// | plain decode | `Exact { acceptable: [] }` | 1, 0, 0 |
/// | verification of `k` proposals | `Exact { acceptable }` | 1, k, 0 |
/// | front pass of depth `k` | `Front(rows)` ([`FrontRows`]) | 1, 0, k |
///
/// A pass of no exact row, or of front rows behind acceptable rows, has no
/// representation: no pass runs one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rows {
    /// The exact row, then `acceptable` proposals (none of a plain decode).
    Exact { acceptable: Vec<TokenId> },
    /// The exact row, then its front rows ([`FrontRows`]), at most the
    /// decode form's rows past the exact row.
    Front(FrontRows),
}

/// A front pass's front rows ([`Rows::Front`]): a front row on each guess
/// of the sequence's chain ([`Guess`]), which the head launch behind the
/// pass before made, then `idle` front rows of no guess, which run on the
/// pending token: a pass granted deeper than the chain reaches
/// ([`crate::FrontGrant::Exactly`]) has them. At least one row: a pass of
/// none is [`Rows::Exact`] ([`Rows::front`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrontRows {
    guesses: Vec<Guess>,
    idle: usize,
}

impl FrontRows {
    /// Its rows, guessed and idle: at least one.
    pub fn rows(&self) -> usize {
        self.guesses.len() + self.idle
    }
}

/// The predictor that made a front row's guess, at the tail of the head
/// launch behind the pass before (`guess_chain`): each stands at its own
/// rate, which the expert ledger prices a front row's names by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GuessKind {
    /// The selection of the pass before's consistent front row at its
    /// depth: the tokens under it are the chain's, the verified one and the
    /// shallower guesses it keeps.
    Chained,
    /// The MTP head's selection on the deepest consistent entry row, its
    /// prediction of the token after that row's selection: one past the
    /// chained guesses.
    Head,
}

impl GuessKind {
    pub const ALL: [Self; 2] = [Self::Chained, Self::Head];

    /// Its word in the device's guesses and step words (guess.seismic's
    /// GUESS_CHAINED, GUESS_HEAD).
    pub const fn word(self) -> i32 {
        use magnitude_kernels::constants::{GUESS_CHAINED, GUESS_HEAD};
        (match self {
            Self::Chained => GUESS_CHAINED,
            Self::Head => GUESS_HEAD,
        }) as i32
    }

    /// The kind a device word names; none of 0, a row that made no guess.
    /// Panics on any other word: the device writes no other.
    pub fn of_word(word: i32) -> Option<Self> {
        match word {
            0 => None,
            word => Some(
                Self::ALL
                    .into_iter()
                    .find(|kind| kind.word() == word)
                    .unwrap_or_else(|| panic!("{word} is no guess kind's word")),
            ),
        }
    }

    /// Its index among [`Self::ALL`], for per-kind tallies.
    pub const fn index(self) -> usize {
        match self {
            Self::Chained => 0,
            Self::Head => 1,
        }
    }
}

/// A guess of the sequence's chain, made behind a decode pass
/// (`guess_chain`): the token a front row of the next pass runs on, and the
/// predictor that made it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Guess {
    pub token: TokenId,
    pub kind: GuessKind,
}

impl Rows {
    /// The pass of a front row on each of `guesses`, then `idle` front rows
    /// of no guess: one exact row where that is no row.
    pub fn front(guesses: Vec<Guess>, idle: usize) -> Self {
        match guesses.len() + idle {
            0 => Self::Exact { acceptable: Vec::new() },
            _ => Self::Front(FrontRows { guesses, idle }),
        }
    }

    /// The proposals whose acceptance may extend the commit.
    pub fn acceptable(&self) -> &[TokenId] {
        match self {
            Self::Exact { acceptable } => acceptable.as_slice(),
            Self::Front(_) => &[],
        }
    }

    /// The rows that are never accepted: its front rows, guessed and idle.
    pub fn front_rows(&self) -> usize {
        match self {
            Self::Exact { .. } => 0,
            Self::Front(front) => front.rows(),
        }
    }

    /// Its rows: the exact row, its acceptable rows, its front rows.
    pub fn row_count(&self) -> usize {
        1 + self.acceptable().len() + self.front_rows()
    }

    /// Its rows' tokens from the sequence's `pending` token, in row order:
    /// the exact row's, then each row ahead's guess, then the pending token
    /// again for each idle front row.
    pub fn tokens(&self, pending: TokenId) -> Vec<TokenId> {
        match self {
            Self::Exact { acceptable } => std::iter::once(pending).chain(acceptable.iter().copied()).collect(),
            Self::Front(front) => std::iter::once(pending)
                .chain(front.guesses.iter().map(|guess| guess.token))
                .chain(std::iter::repeat_n(pending, front.idle))
                .collect(),
        }
    }

    /// Its guessed front rows' guesses, by depth: none of a pass of no
    /// front row, and never an idle row's.
    pub fn guesses(&self) -> &[Guess] {
        match self {
            Self::Exact { .. } => &[],
            Self::Front(front) => &front.guesses,
        }
    }

    /// Each front row's kind as the host builds it, by depth: a guessed
    /// row's predictor, then each idle row's none.
    pub fn front_kinds(&self) -> Vec<FrontKind> {
        let idle = match self {
            Self::Exact { .. } => 0,
            Self::Front(front) => front.idle,
        };
        self.guesses()
            .iter()
            .map(|guess| FrontKind::Host(guess.kind))
            .chain(std::iter::repeat_n(FrontKind::Idle, idle))
            .collect()
    }
}

/// A forward's front row ([`Rows::Front`]): one of its trailing token rows,
/// selected by `select`. Its selection never reaches the outcome's `rows`:
/// it is the next guesses' input alone (`guess_chain`). It computes with only
/// the experts its step holds, and its state is never accepted
/// (`crate::TwoRowDecode`).
/// `kind` is where the predictor that made the guess it runs on is known,
/// which the device carries in the step's words (STEP_GUESSES) for the
/// ledger to price its names by ([`FrontKind`]).
#[derive(Clone, Debug, PartialEq)]
pub struct FrontRow {
    pub select: SelectSpec,
    pub kind: FrontKind,
}

/// Where a front row's guess's kind is known.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrontKind {
    /// On the device alone: a pass queued behind the one that guessed it,
    /// which reads the guess and its kind on the device as it reads its
    /// tokens (`domain::queue::continuation_of`).
    Device,
    /// On the host, which builds the pass from the guess an outcome
    /// returned ([`Outcome::Forward`]'s `guesses`).
    Host(GuessKind),
    /// None: an idle row of a pass the host builds deeper than the chain
    /// reaches ([`Rows::Front`]), which runs on the pending token and whose
    /// names no verdict judges.
    Idle,
}

/// What a pass's round decided of its guessed rows: its rows ahead's
/// tokens in row order (`guesses`, its acceptable rows' then its front
/// rows'), and the sequence's tokens the round selected past its exact row
/// (`truths`: the exact row's selection, then each acceptable row's that
/// the selections before it confirmed). An acceptable row's guess is checked
/// against the selection of the row before it; a front row of depth `d`
/// against the truth at its position, which this round learns only for
/// `d = 1` and a round `d − 1` passes later learns for the rest (the
/// expert ledger joins them by position). A guess stands where it is the
/// selection, a stop token included (acceptance accepts none, but the
/// token is the one selected). A pass that guessed nothing has no guesses.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AheadVerdict {
    guesses: Vec<TokenId>,
    acceptable: usize,
    truths: Vec<TokenId>,
}

impl AheadVerdict {
    /// The verdict of a pass of `rows`, its rows that may be accepted (its
    /// exact row, then its acceptable rows) selecting `samples` in row
    /// order. Panics where fewer rows selected than may be accepted.
    pub fn new(rows: &Rows, samples: &[TokenId]) -> Self {
        // Its idle front rows guessed nothing: no verdict judges them.
        let guesses = match rows {
            Rows::Exact { acceptable } => acceptable.clone(),
            Rows::Front(front) => front.guesses.iter().map(|guess| guess.token).collect(),
        };
        let acceptable = rows.acceptable().len();
        assert!(acceptable < samples.len(), "{acceptable} acceptable rows checked against {} selections", samples.len());
        // Each selection is a truth while the guesses before it stood.
        let confirmed = 1 + guesses[..acceptable].iter().zip(samples).take_while(|(guess, truth)| guess == truth).count();
        Self { guesses, acceptable, truths: samples[..confirmed].to_vec() }
    }

    /// Each guess the round itself checked and the selection it was
    /// checked against, in row order: its acceptable rows', up to the first
    /// that fell, then its first front row's.
    pub fn checked(&self) -> Vec<(TokenId, TokenId)> {
        let checked = self.acceptable.min(self.truths.len());
        let front = usize::from(self.guesses.len() > self.acceptable && self.truths.len() > self.acceptable);
        self.guesses.iter().copied().zip(self.truths.iter().copied()).take(checked + front).collect()
    }

    /// The leading guesses that were acceptable rows; the rest were front
    /// rows.
    pub fn acceptable(&self) -> usize {
        self.acceptable
    }

    /// Its front rows' tokens, by depth.
    pub fn front(&self) -> &[TokenId] {
        &self.guesses[self.acceptable..]
    }

    /// The sequence's tokens it selected past its exact row, in position
    /// order.
    pub fn truths(&self) -> &[TokenId] {
        &self.truths
    }

    /// The leading guesses that the round's own selections confirmed.
    pub fn stood(&self) -> usize {
        self.checked().iter().take_while(|(guess, truth)| guess == truth).count()
    }

    /// Whether the pass guessed nothing.
    pub fn is_empty(&self) -> bool {
        self.guesses.is_empty()
    }
}

/// Conditioning features for [`Operation::Head`]: either host-resident
/// [`FeatureRows`] or a leading `0..rows` slice of an executor-owned on-device
/// [`FeatureRef`] (zero-copy: no D2H read and no H2D upload).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeadFeatures {
    Host(FeatureRows),
    Device {
        features: FeatureRef,
        rows: usize,
    },
}

impl HeadFeatures {
    pub fn rows(&self) -> usize {
        match self {
            Self::Host(rows) => rows.rows(),
            Self::Device { rows, .. } => *rows,
        }
    }

    pub fn host(&self) -> Option<&FeatureRows> {
        match self {
            Self::Host(rows) => Some(rows),
            Self::Device { .. } => None,
        }
    }

    pub fn device(&self) -> Option<(&FeatureRef, usize)> {
        match self {
            Self::Device { features, rows } => Some((features, *rows)),
            Self::Host(_) => None,
        }
    }

    pub fn host_bytes(&self) -> usize {
        match self {
            Self::Host(rows) => rows.bytes().len(),
            Self::Device { .. } => 0,
        }
    }
}

impl From<FeatureRows> for HeadFeatures {
    fn from(rows: FeatureRows) -> Self {
        Self::Host(rows)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Operation {
    Forward {
        request: RequestId,
        kind: WorkKind,
        tokens: Vec<TokenId>,
        position: usize,
        conditioning: Option<ConditioningRef>,
        demand: Demand,
        /// Selection rows in physical token order. Verification supplies one
        /// spec per token; decode supplies one for its sole row; a finishing
        /// prefill supplies one for its final row. Other rows have no spec.
        select: Vec<SelectSpec>,
        committed: usize,
        /// A prompt chunk's entry into a separate drafter, drafted on the
        /// device right behind the chunk from the chunk's own features.
        prime: Option<Priming>,
        /// Its front rows, by depth, of a selecting decode ([`FrontRow`]).
        /// Every row before them is one a decision may accept, selecting
        /// through `select`: `committed` exact rows, then acceptable rows.
        front: Vec<FrontRow>,
        /// The MTP head's entry behind it, of a sequence that decodes in
        /// front passes ([`ChainEntry`]).
        chain: Option<ChainEntry>,
    },
    /// One drafter transaction. The entry rows pair each accepted token with
    /// the target's feature of the preceding row (`conditioning`, in row
    /// order) and are committed to drafter state at `position`. With
    /// proposals, the drafter then drafts one token per `SelectSpec` in its
    /// `form`; its speculative work is never committed.
    Head {
        request: RequestId,
        phase: HeadPhase,
        tokens: Vec<TokenId>,
        conditioning: HeadFeatures,
        position: usize,
        proposals: Vec<SelectSpec>,
        form: DraftForm,
    },
    Encode {
        request: RequestId,
        image: ImageRef,
    },
}

/// A forward's entry into the MTP head that makes the sequence's chain
/// (`crate::two_row`): the head runs on the device right behind the forward,
/// its entry rows pairing the forward's selections with its output features,
/// and at its tail makes the next pass's chain (`guess_chain`). `position` is
/// the head position of its first entry row, the head state's: one past the
/// pass before's, and the closing prompt chunk's one past its drafter
/// entry's rows. Only a selecting decode pass of one exact row (front rows
/// allowed) and the prompt's closing chunk, which primes the head across its
/// boundary, carry one, and each demands the features its entry rows read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainEntry {
    pub position: usize,
}

/// A prompt chunk's drafter entry, committed at `position` with the chunk.
/// A successor-dependent (`boundary`) entry pairs each of its tokens with
/// the feature of the target row before it, so it enters the chunk's pairs
/// in one of two forms ([`Priming::pairs_previous`]): the chunk's every
/// token, its first paired with the boundary feature before the chunk, at
/// the head position before the chunk's; or the chunk's tokens past its
/// first, every pair inside the chunk, at the chunk's position (the first
/// chunk, before which no feature exists, or one whose first token the head
/// entered already). A feature-only entry reads the current chunk directly.
/// Entry rows never exceed the chunk's.
/// `draft_from` is the earliest anchor this history must support, including a
/// continuation from a retained prefix (see [`HeadPhase::Priming`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Priming {
    /// Keep the last target feature independently, entering only token pairs
    /// wholly inside the prefix. Chained MTP heads require this boundary.
    pub boundary: bool,
    pub tokens: Vec<TokenId>,
    pub position: usize,
    pub draft_from: usize,
}

impl Priming {
    /// Whether its first token pairs with the boundary feature before its
    /// chunk of `rows` rows: a successor-dependent entry of the chunk's
    /// every token. Of one of the chunk's tokens past its first, every pair
    /// is inside the chunk.
    pub fn pairs_previous(&self, rows: usize) -> bool {
        self.boundary && self.tokens.len() == rows
    }
}

/// Whether a head transaction primes accepted history or serves generation.
/// Both use the same executor lane, but their physical time has different
/// service accounting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadPhase {
    /// Entering committed history before drafting; the first draft block
    /// cannot anchor before `draft_from`, including after prefix reuse,
    /// so windowed draft histories never read rows before
    /// `draft_from + 1 - window`.
    Priming {
        draft_from: usize,
    },
    Generation,
}

/// How a drafter drafts its proposals after entering its rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DraftForm {
    /// An embedded head (MTP) chains one speculative row per proposal on the
    /// device: each step's selection is the next step's token and its output
    /// feature the next step's conditioning. The chained rows append to head
    /// state as tentative rows (all but the last proposal's).
    Chained,
    /// A separate draft (DFlash, DSpark) reads one non-causal block
    /// `[anchor, mask, …]` after its history, the anchor being the last entry
    /// token, and appends nothing beyond its entry rows.
    Block,
}

/// The draft passes every slot of one head launch shares. A causal head
/// drafts nothing, so it has no selection to pad a drafting slot with;
/// chained drafts pad to the longest; a block's width is physical, so
/// unequal blocks cannot be padded together.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DraftShape {
    Causal,
    Chained,
    Block(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExecutableKind {
    Target,
    Head,
    Encoder,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CommittedClass {
    AllCommitted,
    HasTentative,
}

impl Operation {
    pub const fn request(&self) -> RequestId {
        match self {
            Self::Forward { request, .. }
            | Self::Head { request, .. }
            | Self::Encode { request, .. } => *request,
        }
    }

    pub const fn executable(&self) -> ExecutableKind {
        match self {
            Self::Forward { .. } => ExecutableKind::Target,
            Self::Head { .. } => ExecutableKind::Head,
            Self::Encode { .. } => ExecutableKind::Encoder,
        }
    }

    pub fn demand(&self) -> Demand {
        match self {
            Self::Forward { demand, .. } => *demand,
            Self::Head { proposals, .. } if proposals.is_empty() => Demand::NONE,
            Self::Head { .. } => Demand::SELECT,
            Self::Encode { .. } => Demand::NONE,
        }
    }

    /// Rows this operation advances in its lane's state. A head writes its
    /// entry rows and, chained, one speculative row per proposal step (the
    /// last proposal needs no row of its own).
    pub fn row_count(&self) -> usize {
        match self {
            Self::Forward { tokens, .. } => tokens.len(),
            Self::Head {
                tokens,
                proposals,
                form: DraftForm::Chained,
                ..
            } => tokens.len() + proposals.len().saturating_sub(1),
            Self::Head {
                tokens,
                form: DraftForm::Block,
                ..
            } => tokens.len(),
            Self::Encode { image, .. } => image.patches(),
        }
    }

    /// The leading rows of `row_count` that always commit: a forward's
    /// committed rows, a head's entry rows. The rest are speculative.
    pub fn committed_rows(&self) -> usize {
        match self {
            Self::Forward { committed, .. } => *committed,
            Self::Head { tokens, .. } => tokens.len(),
            Self::Encode { .. } => self.row_count(),
        }
    }

    /// The rows a forward selects: its selection table's and its front
    /// rows'; none of any other operation.
    pub fn selected_rows(&self) -> usize {
        match self {
            Self::Forward { select, front, .. } => select.len() + front.len(),
            Self::Head { .. } | Self::Encode { .. } => 0,
        }
    }

    /// Its entry into the MTP head behind it ([`ChainEntry`]): none of an
    /// operation that chains nothing.
    pub fn chain(&self) -> Option<ChainEntry> {
        match self {
            Self::Forward { chain, .. } => *chain,
            Self::Head { .. } | Self::Encode { .. } => None,
        }
    }

    /// The entry rows of the front head launch behind it ([`ChainEntry`]):
    /// one per row it selects, its exact selection's committed and each
    /// front row's scratch (a decode pass's every row, the closing chunk's
    /// last); none of an operation that chains nothing.
    pub fn entry_rows(&self) -> usize {
        match self.chain() {
            Some(_) => self.selected_rows(),
            None => 0,
        }
    }

    /// Its front rows ([`FrontRow`]), by depth: none of an operation that
    /// runs none.
    pub fn front(&self) -> &[FrontRow] {
        match self {
            Self::Forward { front, .. } => front,
            Self::Head { .. } | Self::Encode { .. } => &[],
        }
    }

    /// Its trailing rows that only run ahead, which no decision accepts: a
    /// forward's front rows ([`Rows`]); none of any other operation.
    pub fn front_rows(&self) -> usize {
        match self {
            Self::Forward { front, .. } => front.len(),
            Self::Head { .. } | Self::Encode { .. } => 0,
        }
    }

    /// The rows past the committed ones that a later decision may still
    /// accept, which a recurrent bank records on its tape: never a front
    /// row (the recurrent contract leaves a row past its tape unrecorded).
    pub fn acceptable_rows(&self) -> usize {
        self.row_count() - self.committed_rows() - self.front_rows()
    }

    pub fn draft_shape(&self) -> Option<DraftShape> {
        match self {
            Self::Head { proposals, .. } if proposals.is_empty() => Some(DraftShape::Causal),
            Self::Head {
                form: DraftForm::Chained,
                ..
            } => Some(DraftShape::Chained),
            Self::Head {
                form: DraftForm::Block,
                proposals,
                ..
            } => Some(DraftShape::Block(proposals.len())),
            Self::Forward { .. } | Self::Encode { .. } => None,
        }
    }

    pub fn commitment(&self) -> CommittedClass {
        match self {
            Self::Forward {
                tokens, committed, ..
            } if *committed < tokens.len() => CommittedClass::HasTentative,
            Self::Head { .. } => CommittedClass::HasTentative,
            _ => CommittedClass::AllCommitted,
        }
    }

    /// Maps a forward's compact selection table back to physical token rows.
    /// Verification maps one-to-one, decode maps its only row, and finishing
    /// prefill maps its sole selection spec to the final token row.
    pub fn selection_for_row(&self, row: usize) -> Option<&SelectSpec> {
        let Self::Forward {
            kind,
            tokens,
            select,
            front,
            ..
        } = self
        else {
            return None;
        };
        // The rows a decision may accept, before the front rows.
        let accepted = tokens.len() - front.len();
        if let Some(depth) = row.checked_sub(accepted) {
            return front.get(depth).map(|front| &front.select);
        }
        match kind {
            WorkKind::Verify => select.get(row),
            WorkKind::Decode if row == 0 => select.first(),
            WorkKind::Prefill if row.checked_add(1) == Some(tokens.len()) => select.first(),
            WorkKind::Decode | WorkKind::Prefill | WorkKind::Replay => None,
        }
    }

    pub fn validate(&self) -> Result<(), OperationError> {
        if self.row_count() == 0 {
            return Err(OperationError::EmptyRows);
        }
        match self {
            Self::Forward {
                kind,
                tokens,
                demand,
                select,
                committed,
                position,
                prime,
                front,
                chain,
                ..
            } => {
                // The rows a decision may accept: `committed` exact rows
                // (at least one), then acceptable rows. Front rows run on
                // the chain the head launch behind the pass before made, so
                // only a chained pass runs them, trailing its one exact row
                // (`chain`, below), within a chained launch's rows.
                let accepted = tokens
                    .len()
                    .checked_sub(front.len())
                    .ok_or(OperationError::FrontRows)?;
                if *committed == 0 || *committed > accepted {
                    return Err(OperationError::CommittedRows {
                        committed: *committed,
                        rows: tokens.len(),
                    });
                }
                if !front.is_empty() {
                    if chain.is_none() || tokens.len() > crate::CHAINED_ROWS {
                        return Err(OperationError::FrontRows);
                    }
                    for front in front {
                        validate_select(&front.select)?;
                    }
                }
                validate_forward_select(*kind, accepted, *demand, select)?;
                if chain.is_some() {
                    let entered = demand.contains(Demand::SELECT)
                        && match kind {
                            WorkKind::Decode => {
                                accepted == 1
                                    && prime.is_none()
                                    && demand.contains(Demand::DEVICE_FEATURES)
                            }
                            WorkKind::Prefill => {
                                prime.as_ref().is_some_and(|prime| prime.boundary)
                                    && front.is_empty()
                                    && demand.contains(Demand::FEATURES)
                            }
                            WorkKind::Verify | WorkKind::Replay => false,
                        };
                    if !entered {
                        return Err(OperationError::Chain);
                    }
                }
                if let Some(prime) = prime {
                    // A successor-dependent entry's pairs are the chunk's,
                    // in one of [`Priming`]'s two forms.
                    let closed_prefix = !prime.boundary
                        || match prime.pairs_previous(tokens.len()) {
                            true => prime.tokens == *tokens && prime.position + 1 == *position,
                            false => prime.tokens == tokens[1..] && prime.position == *position,
                        };
                    if !closed_prefix
                        || *kind != WorkKind::Prefill
                        || !demand.contains(Demand::FEATURES)
                        || (prime.tokens.is_empty() && !prime.boundary)
                        || prime.tokens.len() > tokens.len()
                    {
                        return Err(OperationError::Priming);
                    }
                }
            }
            Self::Head {
                phase,
                tokens,
                conditioning,
                proposals,
                ..
            } => {
                if matches!(phase, HeadPhase::Priming { .. }) && !proposals.is_empty() {
                    return Err(OperationError::PrimingHeadProposals);
                }
                if let HeadFeatures::Device { features, rows } = conditioning {
                    if *rows == 0 || *rows > features.rows() {
                        return Err(OperationError::FeatureSpan {
                            count: features.rows(),
                            rows: *rows,
                        });
                    }
                }
                if conditioning.rows() != tokens.len() {
                    return Err(OperationError::FeatureSpan {
                        count: conditioning.rows(),
                        rows: tokens.len(),
                    });
                }
                for spec in proposals {
                    validate_select(spec)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// A forward's selection table against its `rows` that a decision may
/// accept (its front rows, `front`, select by their own specs, not counted)
/// and demand.
fn validate_forward_select(
    kind: WorkKind,
    rows: usize,
    demand: Demand,
    select: &[SelectSpec],
) -> Result<(), OperationError> {
    if demand.contains(Demand::SELECT) == select.is_empty() {
        return Err(OperationError::SelectionMismatch);
    }
    let expected = match kind {
        WorkKind::Verify => rows,
        WorkKind::Decode if demand.contains(Demand::SELECT) && rows <= 1 => rows,
        WorkKind::Decode if demand.contains(Demand::SELECT) => {
            return Err(OperationError::SelectingDecodeWidth(rows));
        }
        WorkKind::Prefill if demand.contains(Demand::SELECT) => 1,
        WorkKind::Decode | WorkKind::Prefill | WorkKind::Replay => 0,
    };
    if select.len() != expected {
        return Err(OperationError::SelectionRows {
            kind,
            expected,
            actual: select.len(),
        });
    }
    for spec in select {
        validate_select(spec)?;
    }
    Ok(())
}

fn validate_select(select: &SelectSpec) -> Result<(), OperationError> {
    select
        .shaping
        .validate()
        .map_err(|_| OperationError::InvalidShaping)?;
    if select.position > i32::MAX as usize {
        return Err(OperationError::SelectionPosition(select.position));
    }
    if select.mask.as_ref().is_some_and(|mask| mask.is_empty()) {
        return Err(OperationError::EmptyMask);
    }
    match select.history.as_deref() {
        Some(history) if history.len() != 64 => {
            return Err(OperationError::HistoryWidth(history.len()));
        }
        None if select.shaping.uses_history() => {
            return Err(OperationError::MissingHistory);
        }
        _ => {}
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperationError {
    EmptyRows,
    PrimingHeadProposals,
    Priming,
    CommittedRows {
        committed: usize,
        rows: usize,
    },
    SelectionMismatch,
    SelectingDecodeWidth(usize),
    SelectionRows {
        kind: WorkKind,
        expected: usize,
        actual: usize,
    },
    InvalidShaping,
    SelectionPosition(usize),
    EmptyMask,
    MissingHistory,
    HistoryWidth(usize),
    FeatureSpan {
        count: usize,
        rows: usize,
    },
    FeatureRows {
        bytes: usize,
        rows: usize,
    },
    FrontRows,
    Chain,
}

impl fmt::Display for OperationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyRows => formatter.write_str("operation requires at least one row"),
            Self::PrimingHeadProposals => {
                formatter.write_str("priming head cannot draft proposals")
            }
            Self::Priming => formatter.write_str(
                "a drafter entry primes a feature-demanding prompt chunk from at most its rows",
            ),
            Self::CommittedRows { committed, rows } => write!(
                formatter,
                "operation declares {committed} committed rows for a width of {rows}"
            ),
            Self::SelectionMismatch => formatter.write_str(
                "selection demand and selection parameters must either both be present or absent",
            ),
            Self::SelectingDecodeWidth(width) => write!(
                formatter,
                "a selecting Decode must contain one row, not {width}",
            ),
            Self::SelectionRows {
                kind,
                expected,
                actual,
            } => write!(
                formatter,
                "{kind:?} forward has {actual} selection rows; expected {expected}"
            ),
            Self::InvalidShaping => formatter.write_str("selection shaping parameters are invalid"),
            Self::SelectionPosition(position) => write!(
                formatter,
                "selection position {position} exceeds the packed i32 domain"
            ),
            Self::EmptyMask => formatter.write_str("selection mask cannot be empty"),
            Self::MissingHistory => {
                formatter.write_str("history-dependent shaping requires a 64-token history row")
            }
            Self::HistoryWidth(width) => write!(
                formatter,
                "selection history has width {width}; expected 64",
            ),
            Self::FeatureSpan { count, rows } => write!(
                formatter,
                "head conditioning has {count} rows for {rows} token rows",
            ),
            Self::FrontRows => formatter.write_str(
                "front rows trail a chained decode pass's one exact row within a chained launch's rows",
            ),
            Self::Chain => formatter.write_str(
                "a chain entry follows a selecting, feature-demanding decode pass of one exact row, or a prompt's closing chunk priming across its boundary",
            ),
            Self::FeatureRows { bytes, rows } => write!(
                formatter,
                "{bytes} bytes are not {rows} nonempty equal feature rows, or the rows differ in width",
            ),
        }
    }
}

impl std::error::Error for OperationError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selected {
    Token(TokenId),
    Empty,
    NonFinite,
    Declined,
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct RowResult {
    pub selected: Option<Selected>,
    pub features: Option<FeatureRef>,
    pub logits: Option<LogitsRef>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// A forward's rows that a decision may accept (its exact rows, then
    /// its acceptable rows), and the chain the head launch behind it
    /// guessed past its exact selection (`guess_chain`), by depth, of a
    /// decode pass on a load that fronts: empty where it made none, and one
    /// deeper than its front rows at most. No row of `rows` is a front
    /// row's, so nothing a front row computed is ever an output.
    Forward {
        rows: Vec<RowResult>,
        guesses: Vec<Guess>,
    },
    /// One selection per requested proposal, in chain order, with the
    /// drafter's confidence in each, its probability under the logits it was
    /// sampled from, when the drafter samples them: a drafter choosing by
    /// path scores has none to give.
    Head {
        proposals: Vec<Selected>,
        confidence: Option<Vec<f32>>,
    },
    Encode {
        features: FeatureRef,
    },
    /// A forward whose sequence's weights (table rows or experts) could not
    /// be read, in this step or an earlier one (the sequence is poisoned):
    /// its outputs were
    /// discarded, and reconciling it fails the sequence.
    WeightReadFailed(crate::WeightReadError),
    /// A forward its step's expert bank interrupted (a late beat of its
    /// load's pulse: a wait of the step may have ended before its answer),
    /// for `reason`. Its outputs are not to be used, its inputs are whole,
    /// and nothing of it is poisoned: aborting it returns the sequence's
    /// accepted state (a chunk of a chain, the state the chain's reconciled
    /// chunks reached), from which its round is re-run. Reconciling it is an
    /// invariant violation.
    Interrupted { reason: String },
}
