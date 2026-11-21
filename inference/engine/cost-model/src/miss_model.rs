//! The generic miss model: what a bank of a share `s` of a layer's experts
//! misses of what the layer routes, before the load has read any routing of
//! its own, from the layer's `E` and `k` and a single locality assumption.
//!
//! **The ranking it models.** Placement (the executor's `experts::reuse`)
//! holds the `s·E` experts of most expected uses within the group's turnover
//! `τ`. Real MoE decode exhibits strong expert locality — a hot working set
//! carries most selections while the long tail is rarely touched.
//!
//! **The single contained assumption.** Before a load has observed its own
//! routing, each layer's `E` experts are ranked hottest-first with Zipf
//! weights `w_j = j^(−α)` (`j = 1..=E`) at the canonical exponent
//! [`LOCALITY_EXPONENT`] (`α = 1`, Zipf's law). Each expert's single-row
//! selection probability is the unique water-filling cap
//! `p_j = min(1, c · w_j)` with `Σ_j p_j = k`.
//!
//! Every quantity of the model follows directly from `{p_j}`:
//! - [`MissModel::misses`] (`m(s)`): the cold share `1 − (Σ_{j ≤ s·E} p_j)/k`
//!   for a single row, interpolated linearly across fractional slots.
//! - [`MissModel::misses_for`] (`m(s, r)`): the cold share of the distinct
//!   experts routed by `r` independent rows,
//!   `Σ_{j > s·E} (1 − (1 − p_j)^r) / D(r)`. At `r = 1` this equals `m(s)`;
//!   at large `r` (prefill) every expert is routed and `m(s, r) → 1 − s`.
//! - [`MissModel::routed`] (`D(r)`): `Σ_j (1 − (1 − p_j)^r)`, exact at
//!   `D(1) = k`.
//! - [`MissModel::requests`] (`q(s)`): the probability a single row misses at
//!   least one expert when drawing `k` distinct experts without replacement
//!   from a held set of `H = s·E` slots carrying hit share `1 − m(s)`:
//!   `1 − Π_{i < k} ((1 − m)·max(H − i, 0)) / ((1 − m)·max(H − i, 0) + m·H)`.
//!
//! At `α = 0` (`p_j = k/E`), the model reduces identically to the uniform
//! adversarial bound: `m(s) = 1 − s`, `D(r) = E·(1 − (1 − k/E)^r)`, and
//! `q(s) = 1 − Π_{i < k} (H − i)/(E − i)`. A load's own reads replace it
//! layer by layer once it has read there (`LiveDecode`).

/// The single contained locality exponent of the generic miss model:
/// canonical Zipf's law (`w_j = j^(−1)`). `0.0` recovers the uniform
/// adversarial bound.
pub const LOCALITY_EXPONENT: f64 = 1.0;

/// A layer's generic routing model ([`MissModel::of`]): Zipf-concentrated
/// routing of `choices` distinct experts of `experts` per row.
#[derive(Clone, Debug, PartialEq)]
pub struct MissModel {
    experts: f64,
    choices: f64,
    /// Per-expert single-row selection probability `p_j ∈ [0, 1]`, sorted
    /// hottest-first, summing to `choices`.
    probs: Vec<f64>,
    /// Cumulative sum `prefix[h] = Σ_{j < h} probs[j]`, length `experts + 1`,
    /// with `prefix[0] = 0` and `prefix[E] = choices`.
    prefix: Vec<f64>,
}

impl MissModel {
    /// The generic model of a layer of `experts` experts routing `choices` a
    /// row at the default locality exponent ([`LOCALITY_EXPONENT`]).
    pub fn of(experts: u64, choices: u64) -> Self {
        Self::with_locality(experts, choices, LOCALITY_EXPONENT)
    }

    /// The model of a layer of `experts` experts routing `choices` a row at
    /// Zipf exponent `alpha >= 0` (`0.0` is the uniform bound).
    pub fn with_locality(experts: u64, choices: u64, alpha: f64) -> Self {
        assert!(0 < choices && choices <= experts, "{choices} chosen of {experts} experts");
        assert!(alpha.is_finite() && alpha >= 0., "locality exponent must be finite and non-negative: {alpha}");
        let count = usize::try_from(experts).expect("a layer's experts fit in usize");
        let mut weights = Vec::with_capacity(count);
        let mut rem_sum = 0.;
        for rank in 1..=count {
            let w = if alpha == 0. { 1. } else { (rank as f64).powf(-alpha) };
            weights.push(w);
            rem_sum += w;
        }
        // Water-filling cap: since `weights` is non-increasing, experts capped
        // at `p_j = 1` form a prefix `0..cap`.
        let mut cap = 0usize;
        let mut rem_k = choices as f64;
        while cap < count && rem_k > 0. && rem_k * weights[cap] >= rem_sum {
            rem_k -= 1.;
            rem_sum -= weights[cap];
            cap += 1;
        }
        let scale = if rem_sum > 0. && rem_k > 0. { rem_k / rem_sum } else { 0. };
        let mut probs = Vec::with_capacity(count);
        let mut prefix = Vec::with_capacity(count + 1);
        prefix.push(0.);
        let mut acc = 0.;
        for (j, &w) in weights.iter().enumerate() {
            let p = if j < cap { 1. } else { (scale * w).min(1.) };
            probs.push(p);
            acc += p;
            prefix.push(acc);
        }
        if let Some(last) = prefix.last_mut() {
            *last = choices as f64;
        }
        Self { experts: experts as f64, choices: choices as f64, probs, prefix }
    }

    /// `m(s)`: the share of a single row's routed experts that a bank holding
    /// the hottest share `share` of the layer's experts lacks.
    pub fn misses(&self, share: f64) -> f64 {
        assert!((0. ..=1.).contains(&share), "a share of a layer's experts: {share}");
        if share <= 0. {
            return 1.;
        }
        if share >= 1. {
            return 0.;
        }
        let held = share * self.experts;
        let full = (held.floor() as usize).min(self.probs.len());
        let frac = held - full as f64;
        let held_mass = self.prefix[full] + if full < self.probs.len() { frac * self.probs[full] } else { 0. };
        (1. - held_mass / self.choices).clamp(0., 1.)
    }

    /// `m(s, r)`: the share of the distinct experts routed by `rows`
    /// independent rows that a bank holding the hottest share `share` lacks.
    /// Equals [`Self::misses`] at `rows = 1`, and approaches `1 − share` for
    /// large `rows` (prefill) where every expert is routed.
    pub fn misses_for(&self, share: f64, rows: f64) -> f64 {
        assert!((0. ..=1.).contains(&share), "a share of a layer's experts: {share}");
        if share <= 0. {
            return 1.;
        }
        if share >= 1. {
            return 0.;
        }
        if (rows - 1.).abs() <= f64::EPSILON {
            return self.misses(share);
        }
        if !rows.is_finite() || rows <= 0. {
            return 0.;
        }
        let held = share * self.experts;
        let full = (held.floor() as usize).min(self.probs.len());
        let frac = held - full as f64;
        let mut held_routed = 0.;
        let mut total_routed = 0.;
        for (j, &p) in self.probs.iter().enumerate() {
            let u = 1. - (1. - p).powf(rows);
            total_routed += u;
            if j < full {
                held_routed += u;
            } else if j == full {
                held_routed += frac * u;
            }
        }
        if total_routed <= 0. {
            return 0.;
        }
        (1. - held_routed / total_routed).clamp(0., 1.)
    }

    /// `q(s)`: the share of a single row's steps that miss at least one expert
    /// at a bank holding the hottest share `share`. Reduces identically to the
    /// hypergeometric `1 − Π_{i < k} (H − i)/(E − i)` at `α = 0`.
    pub fn requests(&self, share: f64) -> f64 {
        assert!((0. ..=1.).contains(&share), "a share of a layer's experts: {share}");
        if share <= 0. {
            return 1.;
        }
        if share >= 1. {
            return 0.;
        }
        let held = share * self.experts;
        let misses = self.misses(share);
        let hits = 1. - misses;
        let all_held = (0..self.choices as u64)
            .map(|i| {
                let rem = (held - i as f64).max(0.);
                let num = hits * rem;
                let den = num + misses * held;
                if den <= 0. { 0. } else { num / den }
            })
            .product::<f64>();
        (1. - all_held).clamp(0., 1.)
    }

    /// `D(r) = Σ_j (1 − (1 − p_j)^r)`: the expected distinct experts `rows`
    /// independent rows route under this distribution.
    pub fn routed(&self, rows: f64) -> f64 {
        if !rows.is_finite() || rows <= 0. {
            return 0.;
        }
        if (rows - 1.).abs() <= f64::EPSILON {
            return self.choices;
        }
        self.probs.iter().map(|&p| 1. - (1. - p).powf(rows)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A larger bank never misses more, none misses everything, a whole one
    /// nothing; and so for the steps that miss at all.
    #[test]
    fn the_curve_falls_from_all_to_none() {
        for model in [MissModel::of(256, 8), MissModel::of(512, 10), MissModel::with_locality(256, 8, 0.)] {
            assert_eq!((model.misses(0.), model.misses(1.)), (1., 0.));
            assert_eq!((model.requests(0.), model.requests(1.)), (1., 0.));
            assert_eq!((model.misses_for(0., 4.), model.misses_for(1., 4.)), (1., 0.));
            let shares = (0..=256).map(|slots| slots as f64 / 256.).collect::<Vec<_>>();
            assert!(shares.windows(2).all(|pair| model.misses(pair[1]) <= model.misses(pair[0]) + 1e-12));
            assert!(shares.windows(2).all(|pair| model.requests(pair[1]) <= model.requests(pair[0]) + 1e-12));
            assert!(shares.windows(2).all(|pair| model.misses_for(pair[1], 4.) <= model.misses_for(pair[0], 4.) + 1e-12));
        }
    }

    /// At `α = 0`, a row of 2 of 4 experts beside a bank of 2 matches the exact
    /// hypergeometric `5/6`; with locality, the hottest 2 carry more mass so
    /// requests are strictly below `5/6`; a row of every expert misses any bank
    /// short of all regardless of locality.
    #[test]
    fn a_row_misses_at_all_unless_every_choice_is_held() {
        assert!((MissModel::with_locality(4, 2, 0.).requests(0.5) - 5. / 6.).abs() < 1e-12);
        assert!(MissModel::of(4, 2).requests(0.5) < 5. / 6.);
        assert_eq!(MissModel::of(4, 4).requests(0.75), 1., "a row of every expert misses any bank short of all");
        assert_eq!(MissModel::of(512, 10).requests(9. / 512.), 1., "fewer than k slots cannot hold k distinct choices");
    }

    /// At `α = 0`, any ranking misses at most the uniform bound `1 − s`; with
    /// locality (`α > 0`), the Zipf model strictly improves on the uniform
    /// bound for every interior share `0 < s < 1`.
    #[test]
    fn locality_improves_on_the_uniform_bound() {
        let uniform = MissModel::with_locality(512, 10, 0.);
        let zipf = MissModel::of(512, 10);
        for slots in 1..512 {
            let s = slots as f64 / 512.;
            assert!((uniform.misses(s) - (1. - s)).abs() < 1e-12);
            assert!(zipf.misses(s) < uniform.misses(s), "at {slots}/512: zipf={} uniform={}", zipf.misses(s), uniform.misses(s));
            assert!(zipf.requests(s) <= uniform.requests(s) + 1e-12);
        }
        // At 70% residency on E=512, k=10, Zipf α=1 gives ~5.5% misses (vs 30% uniform).
        let m70 = zipf.misses(0.70);
        assert!(m70 > 0.03 && m70 < 0.10, "m(0.70) = {m70}");
    }

    /// A single row routes `k`; independent rows at most `E`; multi-row
    /// `misses_for` equals `misses` at 1 row and converges to `1 − s` at large
    /// prefill row counts.
    #[test]
    fn rows_route_their_union_and_multi_row_misses_converge() {
        let uniform = MissModel::with_locality(256, 8, 0.);
        let zipf = MissModel::of(256, 8);
        assert!((uniform.routed(1.) - 8.).abs() < 1e-9);
        assert!((zipf.routed(1.) - 8.).abs() < 1e-9);
        let expected_uniform = 256. * (1. - (1. - 8. / 256f64).powi(8));
        assert!((uniform.routed(8.) - expected_uniform).abs() < 1e-9);
        assert!(zipf.routed(8.) > 8. && zipf.routed(8.) < expected_uniform);
        assert_eq!(MissModel::of(4, 4).routed(3.), 4.);
        // At 1 row, `misses_for` equals `misses`.
        assert!((zipf.misses_for(0.5, 1.) - zipf.misses(0.5)).abs() < 1e-12);
        // At large prefill rows, every expert is routed so cold share -> 1 - s.
        let prefill_miss = MissModel::of(32, 4).misses_for(0.25, 512.);
        assert!((prefill_miss - 0.75).abs() < 1e-4, "prefill_miss={prefill_miss}");
    }
}
