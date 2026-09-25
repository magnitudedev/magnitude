//! Reductions in a defined order.
//!
//! Sums accumulate in [`LANES`] independent lanes, element `i` into lane
//! `i % LANES` in index order, and combine the lanes by a fixed tree. The
//! order depends only on the element count, never on the tier, so every tier
//! produces the same bits; the loops vectorize to the tier's registers.

/// Lanes of every defined-order sum.
pub const LANES: usize = 8;

/// The lanes of one sum combined in the fixed tree
/// `((l0 + l4) + (l2 + l6)) + ((l1 + l5) + (l3 + l7))`.
#[inline(always)]
pub fn combine(lanes: [f32; LANES]) -> f32 {
    ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6])) + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]))
}

/// `sum(values)`.
#[inline(always)]
pub fn sum(values: &[f32]) -> f32 {
    let mut lanes = [0.0f32; LANES];
    let chunks = values.chunks_exact(LANES);
    let tail = chunks.remainder();
    for chunk in chunks {
        for lane in 0..LANES {
            lanes[lane] += chunk[lane];
        }
    }
    for (lane, value) in tail.iter().enumerate() {
        lanes[lane] += value;
    }
    combine(lanes)
}

/// `sum(values[i]^2)`, each square fused into its lane.
#[inline(always)]
pub fn sum_squares(values: &[f32]) -> f32 {
    let mut lanes = [0.0f32; LANES];
    let chunks = values.chunks_exact(LANES);
    let tail = chunks.remainder();
    for chunk in chunks {
        for lane in 0..LANES {
            lanes[lane] = chunk[lane].mul_add(chunk[lane], lanes[lane]);
        }
    }
    for (lane, value) in tail.iter().enumerate() {
        lanes[lane] = value.mul_add(*value, lanes[lane]);
    }
    combine(lanes)
}

/// `sum(a[i] * b[i])` over the common length, each product fused into its
/// lane.
#[inline(always)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let length = a.len().min(b.len());
    let (a, b) = (&a[..length], &b[..length]);
    let mut lanes = [0.0f32; LANES];
    let full = length / LANES * LANES;
    for (a, b) in a[..full].chunks_exact(LANES).zip(b[..full].chunks_exact(LANES)) {
        for lane in 0..LANES {
            lanes[lane] = a[lane].mul_add(b[lane], lanes[lane]);
        }
    }
    for (lane, (a, b)) in a[full..].iter().zip(&b[full..]).enumerate() {
        lanes[lane] = a.mul_add(*b, lanes[lane]);
    }
    combine(lanes)
}

/// The largest value; NaN propagates.
#[inline(always)]
pub fn max(values: &[f32]) -> f32 {
    values.iter().fold(f32::NEG_INFINITY, |best, value| {
        if value.is_nan() || best.is_nan() {
            f32::NAN
        } else {
            best.max(*value)
        }
    })
}

/// The index of the largest value, the lowest index among ties; NaN is
/// never selected unless every value is NaN.
#[inline(always)]
pub fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] || values[best].is_nan() && !value.is_nan() {
            best = index;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_follow_the_lane_order() {
        let values = (0..19).map(|i| 1.0 + i as f32 * 1e-7).collect::<Vec<_>>();
        let mut lanes = [0.0f32; LANES];
        for (index, value) in values.iter().enumerate() {
            lanes[index % LANES] += value;
        }
        assert_eq!(sum(&values).to_bits(), combine(lanes).to_bits());
        assert_eq!(dot(&values, &values).to_bits(), sum_squares(&values).to_bits());
    }

    #[test]
    fn argmax_takes_the_first_of_ties() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[f32::NAN, 1.0]), 1);
    }
}
