//! Checked geometry for contiguous, order-preserving tensor views.
pub fn reshape_strides(shape: &[i64], strides: &[i64], target: &[i64]) -> Result<Vec<i64>, String> {
    if shape.len() != strides.len()
        || shape.iter().chain(target).any(|n| *n < 0)
        || target.is_empty()
    {
        return Err("reshape requires nonnegative dimensions and a nonempty target rank".into());
    }
    let product = |dims: &[i64]| {
        dims.iter().try_fold(1i64, |n, d| {
            n.checked_mul(*d)
                .ok_or_else(|| "reshape size overflow".to_string())
        })
    };
    let count = product(shape)?;
    if count != product(target)? {
        return Err("reshape must preserve element count".into());
    }
    let mut expected = 1i64;
    if count != 0 {
        for (&dim, &stride) in shape.iter().zip(strides).rev() {
            if dim > 1 && stride != expected {
                return Err("reshape requires contiguous row-major storage".into());
            }
            expected = expected.checked_mul(dim).ok_or("reshape stride overflow")?;
        }
    }
    row_major_strides(target)
}

pub fn row_major_strides(target: &[i64]) -> Result<Vec<i64>, String> {
    if target.iter().any(|n| *n < 0) {
        return Err("negative tensor extent".into());
    }
    let mut result = vec![1; target.len()];
    let mut stride = 1i64;
    for (axis, dim) in target.iter().enumerate().rev() {
        result[axis] = stride;
        stride = stride.checked_mul(*dim).ok_or("reshape stride overflow")?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn contiguous_reshape_preserves_linear_order() {
        for a in 0..8 {
            for b in 1..8 {
                let shape = [a, 1, b];
                let strides = [b, 999, 1]; // Singleton-axis stride cannot affect an address.
                let target = [b, a];
                let got = reshape_strides(&shape, &strides, &target).unwrap();
                for i in 0..a * b {
                    assert_eq!((i / a) * got[0] + (i % a) * got[1], i);
                }
            }
        }
        assert!(reshape_strides(&[3, 2], &[4, 1], &[6]).is_err());
        assert!(reshape_strides(&[2, 3], &[1, 2], &[6]).is_err());
        assert!(reshape_strides(&[6], &[1], &[5]).is_err());
        assert!(reshape_strides(&[6], &[1], &[-6]).is_err());
        assert!(reshape_strides(&[i64::MAX, 2], &[2, 1], &[2, i64::MAX]).is_err());
    }
}
