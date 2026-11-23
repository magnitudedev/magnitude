//! A token's terms: the one statement of what each resource a streamed
//! decode token uses is busy for, on any memory topology
//! (streamed-moe-decode-dynamics.md §1, §3, §6).
//!
//! At decode's rows every resource is a bandwidth, so a token is the busy
//! seconds of each:
//!
//! - **compute** (`C(1 + k)`): the GPU's pass over the memory it computes
//!   from, its exact and front rows;
//! - **paths**, each the experts that cross it times one's time on it:
//!   **drive** (the reads its misses needed, read ahead or on demand) and,
//!   beside a host tier, the host-to-device **link** (the experts the GPU
//!   read over it in place, and the moves, promotions, that copied experts
//!   into its memory);
//! - **in-place tiers**, each the work a slower tier's own processor does
//!   from its memory: the **cpu** (the experts its site computed from host
//!   memory), and host **memory** itself, which the CPU's reads, the link's
//!   reads and the drive's landings all share. On unified memory the CPU
//!   and the GPU read one memory, whose bandwidth `C` already counts: there
//!   is no link and no second memory, and those terms are none.
//!
//! With every miss named early enough the terms overlap and the slowest
//! sets the pace, `T = max(C(1 + k), every path, every in-place tier)`
//! ([`Terms::time`]); a term the token waited on in series shows in the
//! pass's own time. Every price of a streamed decode reads these terms:
//! the regime's depth weighs the GPU's term against every other one
//! ([`Terms::each_path`], `crate::Regime`).

/// A physical bandwidth or execution contention domain shared across tiers and
/// paths on a machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ContentionDomain {
    Compute,
    Drive,
    Link,
    HostMemory,
    Cpu,
}

/// Which physical streaming path an admission or scheduling decision governs:
/// - `Drive`: storage -> host RAM (or unified memory bank);
/// - `Link`: host RAM / peer -> device VRAM.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PathKind {
    #[default]
    Drive,
    Link,
}

impl PathKind {
    /// The transport contention domain owned by this path.
    pub const fn transport_domain(self) -> ContentionDomain {
        match self {
            Self::Drive => ContentionDomain::Drive,
            Self::Link => ContentionDomain::Link,
        }
    }

    /// Physical contention domains shared with other paths or in-place compute
    /// (e.g. Host DDR5 on discrete hosts, whose term is `0.0` on unified memory).
    pub const fn shared_domains(self) -> &'static [ContentionDomain] {
        match self {
            Self::Drive | Self::Link => &[ContentionDomain::HostMemory],
        }
    }
}

/// One token's terms: each resource's busy seconds a token
/// (module documentation). On unified memory `link`, `cpu` and `memory` are
/// none.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Terms {
    pub compute: f64,
    pub drive: f64,
    pub link: f64,
    pub cpu: f64,
    pub memory: f64,
}

/// Alias naming [`Terms`] as a token's per-domain resource busy seconds.
pub type ResourceTerms = Terms;

impl Terms {
    /// Busy seconds of physical contention domain `domain`.
    pub fn domain(&self, domain: ContentionDomain) -> f64 {
        match domain {
            ContentionDomain::Compute => self.compute,
            ContentionDomain::Drive => self.drive,
            ContentionDomain::Link => self.link,
            ContentionDomain::HostMemory => self.memory,
            ContentionDomain::Cpu => self.cpu,
        }
    }

    /// The slowest term among `path`'s own transport domain and any shared
    /// physical contention domains (e.g. Host DDR5).
    pub fn path_term(&self, path: PathKind) -> f64 {
        let own = self.domain(path.transport_domain());
        path.shared_domains().iter().fold(own, |acc, &d| acc.max(self.domain(d)))
    }

    /// Whether `path` (via its own transport domain or a shared contention
    /// domain such as Host DDR5) is a binding resource:
    /// `path_term > 0 && path_term >= time()`.
    pub fn binds_path(&self, path: PathKind) -> bool {
        let peak = self.time();
        peak > 0.0 && self.path_term(path) >= peak
    }

    /// Every term besides the GPU's pass, each path and in-place tier, the
    /// drive's first: what the GPU's pass overlaps with foresight.
    pub fn each_path(&self) -> [f64; 4] {
        [self.drive, self.link, self.cpu, self.memory]
    }

    /// The busiest term besides the GPU's pass ([`Self::each_path`]).
    pub fn paths(&self) -> f64 {
        self.each_path().into_iter().fold(0., f64::max)
    }

    /// `T`: the token's time with every term overlapped, the slowest.
    pub fn time(&self) -> f64 {
        self.compute.max(self.paths())
    }

    /// `self + sign · terms`, term by term.
    pub fn add(&mut self, terms: &Self, sign: f64) {
        self.compute += sign * terms.compute;
        self.drive += sign * terms.drive;
        self.link += sign * terms.link;
        self.cpu += sign * terms.cpu;
        self.memory += sign * terms.memory;
    }

    /// Every term over `count`: a mean of `count` tokens' sums.
    pub fn over(&self, count: f64) -> Self {
        Self { compute: self.compute / count, drive: self.drive / count, link: self.link / count, cpu: self.cpu / count, memory: self.memory / count }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three regimes a topology can be in, each its binding term. On
    /// unified memory the drive or the GPU binds; on a discrete host whose
    /// RAM holds the model the drive is idle and the link, the CPU or the
    /// GPU binds; with RAM short of the model the drive binds again. The
    /// GPU's pass is overlapped by the rest exactly where it is the
    /// slowest.
    #[test]
    fn the_slowest_term_sets_the_pace_on_every_topology() {
        let unified = Terms { compute: 0.033, drive: 0.150, ..Terms::default() };
        assert_eq!((unified.paths(), unified.time()), (0.150, 0.150));
        let resident = Terms { drive: 0.0015, ..unified };
        assert_eq!(resident.time(), 0.033, "a drive under the pass is hidden");
        let fits = Terms { compute: 0.020, drive: 0., link: 0.045, cpu: 0.030, memory: 0.012 };
        assert_eq!(fits.time(), 0.045, "RAM holds the model: the link binds");
        let short = Terms { drive: 0.090, ..fits };
        assert_eq!(short.time(), 0.090, "RAM short of the model: the drive binds");
        // The regime's depth follows the binding term: every term a miss's
        // share of it (one miss a step), the misses falling a 40th each
        // depth foreseen, against code's pass (C(1..8)): the most front rows
        // whose pass the busiest other term outweighs.
        const CODE: [f64; 8] = [0.0334, 0.0356, 0.0375, 0.0396, 0.0419, 0.0445, 0.0475, 0.0508];
        let depth = |terms: Terms| {
            let paths = terms.each_path().map(|seconds| crate::PathTerm { seconds, wasted: crate::Waste { front: vec![None; 8], verify: vec![None; 8] } }).to_vec();
            crate::Regime { resident: CODE.to_vec(), needed: crate::NeededByForm::both((0..8).map(|k| crate::Needed::Executed(1. - k as f64 / 40.)).collect()), stand: None, accepts: vec![None; 7], stall: None, overhead: None, paths, routes: crate::VerifyRoutes::default() }.depth()
        };
        assert_eq!((depth(unified), depth(resident)), (7, 0), "unified: the drive, else the GPU");
        assert_eq!((depth(fits), depth(short)), (3, 7), "discrete: the link, else the drive");
        let mut sum = Terms::default();
        sum.add(&fits, 1.);
        sum.add(&short, 1.);
        sum.add(&fits, -1.);
        assert_eq!(sum.over(1.), short);
    }
}
