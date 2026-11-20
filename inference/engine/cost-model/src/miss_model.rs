//! The generic miss model: what a bank of a share `s` of a layer's experts
//! misses of what the layer routes, before the load has read any routing of
//! its own, from the layer's `E` and `k` alone.
//!
//! **The ranking it models.** Placement (the executor's `experts::reuse`)
//! holds the `s·E` experts of most expected uses within the group's turnover
//! `τ`, the steps the pool takes to replace its records at its miss rate:
//! `τ = s·E / (m·D)`, `D` the distinct experts a step routes. The share it
//! misses is therefore a fixed point, `m = F(s, τ(m))`, `F(s, τ)` the share of
//! a step's routed experts outside the `s·E` of most expected uses within `τ`.
//!
//! **What is known at load.** `E`, `k` and the layers. No routing of the model
//! has been read: a load stores nothing it counts, so nothing measured of how
//! its rows choose exists before it decodes, and the routing is unknown past
//! each row choosing `k` distinct experts of `E`.
//!
//! **The bound, and the case it is exact for.** Whatever the routing, the
//! `s·E` experts of most expected uses within any `τ` hold at least the share
//! `s` of the uses expected (the top `s` of a set hold at least `s` of its
//! sum), so a ranking whose expectations are calibrated against what followed,
//! as placement's are, misses at most `F(s, τ) ≤ 1 − s` at every `τ`, and so at
//! the fixed point. The bound is attained exactly where routing tells nothing
//! of the future, each row's `k` chosen uniformly and independently: every
//! expert's expected uses within any `τ` are `τ·k/E`, the held set is
//! independent of the step's choices, and each routed expert misses with
//! `1 − s`. There `F(s, τ) = 1 − s` for every `τ`, so the fixed point is
//! solved in closed form, `m(s) = 1 − s`, whatever turnover it implies.
//!
//! The model is that uninformed case throughout: its misses are the bound, and
//! its distinct experts ([`MissModel::routed`]) and requests
//! ([`MissModel::requests`]) are that same routing's, exact for it, so the
//! three describe one routing. A load's own reads replace it layer by layer
//! once it has read there (`LiveDecode`).

/// A layer's generic model ([`MissModel::of`]): uniform, independent routing
/// of `choices` of `experts` a row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MissModel {
    experts: f64,
    choices: f64,
}

impl MissModel {
    /// The model of a layer of `experts` experts routing `choices` a row.
    pub fn of(experts: u64, choices: u64) -> Self {
        assert!(0 < choices && choices <= experts, "{choices} chosen of {experts} experts");
        Self { experts: experts as f64, choices: choices as f64 }
    }

    /// `m(s) = 1 − s`: the share of what a step routes that a bank of a share
    /// `share` of the layer's experts lacks, the ranking's fixed point at the
    /// bound (module docs).
    pub fn misses(&self, share: f64) -> f64 {
        assert!((0. ..=1.).contains(&share), "a share of a layer's experts: {share}");
        1. - share
    }

    /// `q(s) = 1 − Π_{i<k} (s·E − i)/(E − i)`: the share of a single row's
    /// steps that miss at all at a bank of a share `share`, the row's `k`
    /// distinct choices all held with the hypergeometric chance a uniform
    /// draw of `k` of `E` falls within the `s·E` held.
    pub fn requests(&self, share: f64) -> f64 {
        assert!((0. ..=1.).contains(&share), "a share of a layer's experts: {share}");
        let held = share * self.experts;
        let all_held = (0..self.choices as u64).map(|i| ((held - i as f64) / (self.experts - i as f64)).max(0.)).product::<f64>();
        1. - all_held
    }

    /// `D(r) = E·(1 − (1 − k/E)^r)`: the distinct experts `rows` independent
    /// rows route, each missing a given expert with `1 − k/E`. Of any routing
    /// whose rows choose independently, the most (`1 − (1 − p)^r` is concave
    /// in each expert's `p`, and the `p` sum to `k`).
    pub fn routed(&self, rows: f64) -> f64 {
        self.experts * (1. - (1. - self.choices / self.experts).powf(rows))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A larger bank never misses more, none misses everything, a whole one
    /// nothing; and so for the steps that miss at all.
    #[test]
    fn the_curve_falls_from_all_to_none() {
        let model = MissModel::of(256, 8);
        assert_eq!((model.misses(0.), model.misses(1.)), (1., 0.));
        assert_eq!((model.requests(0.), model.requests(1.)), (1., 0.));
        let shares = (0..=256).map(|slots| slots as f64 / 256.).collect::<Vec<_>>();
        assert!(shares.windows(2).all(|pair| model.misses(pair[1]) <= model.misses(pair[0])));
        assert!(shares.windows(2).all(|pair| model.requests(pair[1]) <= model.requests(pair[0])));
    }

    /// A row of 2 of 4 experts beside a bank of 2: one of the 6 pairs it may
    /// choose is held whole.
    #[test]
    fn a_row_misses_at_all_unless_every_choice_is_held() {
        assert!((MissModel::of(4, 2).requests(0.5) - 5. / 6.).abs() < 1e-12);
        assert_eq!(MissModel::of(4, 4).requests(0.75), 1., "a row of every expert misses any bank short of all");
    }

    /// The ranking of any routing it knows misses no more than the bound: the
    /// most used `s·E` experts carry at least `s` of the uses.
    #[test]
    fn a_ranking_that_knows_its_routing_misses_at_most_the_bound() {
        let model = MissModel::of(8, 2);
        let routings: [[f64; 8]; 3] = [[0.25; 8], [1., 0.5, 0.2, 0.1, 0.1, 0.05, 0.03, 0.02], [0.9, 0.9, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03]];
        for routing in routings {
            for held in 0..=8 {
                let caught = routing[..held].iter().sum::<f64>() / 2.;
                assert!(1. - caught <= model.misses(held as f64 / 8.) + 1e-12, "{routing:?} at {held}");
            }
        }
    }

    /// A single row routes `k`; independent rows at most `E`.
    #[test]
    fn rows_route_their_union() {
        let model = MissModel::of(256, 8);
        assert!((model.routed(1.) - 8.).abs() < 1e-9, "{}", model.routed(1.));
        let uniform = 256. * (1. - (1. - 8. / 256f64).powi(8));
        assert!((model.routed(8.) - uniform).abs() < 1e-9, "{} of {uniform}", model.routed(8.));
        assert_eq!(MissModel::of(4, 4).routed(3.), 4.);
    }
}
