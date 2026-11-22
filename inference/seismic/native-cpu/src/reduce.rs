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
    ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6]))
        + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]))
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
    for (a, b) in a[..full]
        .chunks_exact(LANES)
        .zip(b[..full].chunks_exact(LANES))
    {
        for lane in 0..LANES {
            lanes[lane] = a[lane].mul_add(b[lane], lanes[lane]);
        }
    }
    for (lane, (a, b)) in a[full..].iter().zip(&b[full..]).enumerate() {
        lanes[lane] = a.mul_add(*b, lanes[lane]);
    }
    combine(lanes)
}

#[inline(always)]
fn max_lanes(lanes: &[f32; LANES]) -> f32 {
    let m0 = lanes[0].max(lanes[4]);
    let m1 = lanes[1].max(lanes[5]);
    let m2 = lanes[2].max(lanes[6]);
    let m3 = lanes[3].max(lanes[7]);
    m0.max(m2).max(m1.max(m3))
}

/// The largest value; NaN propagates.
#[inline(always)]
pub fn max(values: &[f32]) -> f32 {
    let mut lanes = [f32::NEG_INFINITY; LANES];
    let mut nan = [false; LANES];
    let chunks = values.chunks_exact(LANES);
    let tail = chunks.remainder();
    for chunk in chunks {
        for lane in 0..LANES {
            let value = chunk[lane];
            lanes[lane] = lanes[lane].max(value);
            nan[lane] |= value.is_nan();
        }
    }
    for (lane, &value) in tail.iter().enumerate() {
        lanes[lane] = lanes[lane].max(value);
        nan[lane] |= value.is_nan();
    }
    if nan.iter().any(|&has_nan| has_nan) {
        f32::NAN
    } else {
        max_lanes(&lanes)
    }
}

/// The index of the largest value, the lowest index among ties; NaN is
/// never selected unless every value is NaN.
#[inline(always)]
pub fn argmax(values: &[f32]) -> usize {
    let Some(&first) = values.first() else {
        return 0;
    };
    let mut best = 0;
    let mut best_val = first;
    let chunks = values.chunks_exact(LANES);
    let tail = chunks.remainder();
    let tail_offset = values.len() - tail.len();
    for (chunk_idx, chunk) in chunks.enumerate() {
        let chunk: &[f32; LANES] = chunk.try_into().unwrap();
        let chunk_max = max_lanes(chunk);
        if chunk_max > best_val || (best_val.is_nan() && !chunk_max.is_nan()) {
            let base = chunk_idx * LANES;
            for (lane, &value) in chunk.iter().enumerate() {
                if value > best_val || (best_val.is_nan() && !value.is_nan()) {
                    best = base + lane;
                    best_val = value;
                }
            }
        }
    }
    for (offset, &value) in tail.iter().enumerate() {
        if value > best_val || (best_val.is_nan() && !value.is_nan()) {
            best = tail_offset + offset;
            best_val = value;
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
        assert_eq!(
            dot(&values, &values).to_bits(),
            sum_squares(&values).to_bits()
        );
    }

    #[test]
    fn max_propagates_nan_and_preserves_signed_zero() {
        assert_eq!(max(&[]), f32::NEG_INFINITY);
        assert!(max(&[f32::NAN]).is_nan());
        assert!(max(&[f32::NAN; 19]).is_nan());

        // NaN at every position (head, mid, tail across all lanes) propagates.
        for nan_pos in 0..19 {
            let mut values = (0..19).map(|i| i as f32 - 5.0).collect::<Vec<_>>();
            values[nan_pos] = f32::NAN;
            assert!(max(&values).is_nan(), "NaN at index {nan_pos} did not propagate");
        }

        // Normal max across chunks and tail.
        let values = (0..19).map(|i| i as f32 - 5.0).collect::<Vec<_>>();
        assert_eq!(max(&values), 13.0);

        // Signed-zero semantics: -0.0 preserved when no +0.0, +0.0 wins over -0.0.
        assert_eq!(max(&[-0.0]).to_bits(), (-0.0f32).to_bits());
        assert_eq!(max(&[-0.0; 11]).to_bits(), (-0.0f32).to_bits());
        assert_eq!(max(&[-0.0, 0.0]).to_bits(), (0.0f32).to_bits());
        assert_eq!(max(&[0.0, -0.0]).to_bits(), (0.0f32).to_bits());
        let mut across_chunks = [-0.0f32; 11];
        across_chunks[9] = 0.0;
        assert_eq!(max(&across_chunks).to_bits(), (0.0f32).to_bits());
    }

    #[test]
    fn argmax_takes_the_first_of_ties() {
        assert_eq!(argmax(&[]), 0);
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[f32::NAN, 1.0]), 1);

        // All-NaN slices return index 0.
        assert_eq!(argmax(&[f32::NAN; 3]), 0);
        assert_eq!(argmax(&[f32::NAN; 19]), 0);

        // Leading NaNs spanning full chunks still select a later non-NaN (even -inf).
        let mut leading_nan_chunk = [f32::NAN; 19];
        leading_nan_chunk[10] = f32::NEG_INFINITY;
        assert_eq!(argmax(&leading_nan_chunk), 10);

        let mut leading_nan_tail = [f32::NAN; 19];
        leading_nan_tail[17] = f32::NEG_INFINITY;
        assert_eq!(argmax(&leading_nan_tail), 17);

        // Ties across chunk boundaries pick the lowest index.
        let mut boundary_tie = [0.0f32; 19];
        boundary_tie[7] = 5.0;
        boundary_tie[8] = 5.0;
        boundary_tie[18] = 5.0;
        assert_eq!(argmax(&boundary_tie), 7);

        // Signed-zero tie across chunk boundary keeps the lowest index.
        let mut zero_tie = [-1.0f32; 19];
        zero_tie[7] = -0.0;
        zero_tie[8] = 0.0;
        assert_eq!(argmax(&zero_tie), 7);

        // NaNs after the maximum in later chunks or tail are never selected.
        let mut mid_and_tail_nan = [1.0f32; 19];
        mid_and_tail_nan[3] = 4.0;
        mid_and_tail_nan[9] = f32::NAN;
        mid_and_tail_nan[18] = f32::NAN;
        assert_eq!(argmax(&mid_and_tail_nan), 3);

        // Maximum strictly in the tail.
        let mut tail_max = [1.0f32; 19];
        tail_max[18] = 9.0;
        assert_eq!(argmax(&tail_max), 18);
    }
}
