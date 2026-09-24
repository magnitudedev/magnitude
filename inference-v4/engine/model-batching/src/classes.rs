use crate::Demand;
use std::fmt;

pub const MAX_CLASS_ROWS: usize = 512;
/// The history-segment ladder ends at the most segments an accepted history
/// may have before it needs compaction.
pub const MAX_CLASS_SEGMENTS: usize = magnitude_model_state::MAX_VISIBLE_SEGMENTS;

/// Row classes at and below this bound are powers of two (decode, verify and
/// concurrency rows); above it they are multiples of [`PREFILL_ROW_QUANTUM`].
pub const SMALL_CLASS_ROWS: usize = 32;
/// Prefill row classes are multiples of this quantum up to [`MAX_CLASS_ROWS`].
pub const PREFILL_ROW_QUANTUM: usize = 64;

/// The row class covering `actual` rows: the next power of two up to
/// [`SMALL_CLASS_ROWS`], otherwise the next multiple of
/// [`PREFILL_ROW_QUANTUM`]. `None` for zero rows or rows beyond
/// [`MAX_CLASS_ROWS`].
pub const fn row_class(actual: usize) -> Option<usize> {
    if actual == 0 || actual > MAX_CLASS_ROWS {
        return None;
    }
    if actual <= SMALL_CLASS_ROWS {
        return Some(actual.next_power_of_two());
    }
    Some(actual.div_ceil(PREFILL_ROW_QUANTUM) * PREFILL_ROW_QUANTUM)
}

/// Every row class a runtime admitting at most `limit` rows can launch, in
/// ascending order: the ladder up to and including the class covering
/// `limit`. Empty when `limit` is zero or exceeds [`MAX_CLASS_ROWS`].
pub fn row_classes(limit: usize) -> Vec<usize> {
    let Some(last) = row_class(limit) else {
        return Vec::new();
    };
    let mut classes = Vec::new();
    let mut rows = 1;
    while rows <= last {
        classes.push(rows);
        rows = if rows < SMALL_CLASS_ROWS {
            rows * 2
        } else if rows == SMALL_CLASS_ROWS {
            PREFILL_ROW_QUANTUM
        } else {
            rows + PREFILL_ROW_QUANTUM
        };
    }
    classes
}

/// Physical preparation class shared by native and planned execution paths.
///
/// The native path uses this as launch shape and qualification data; planned
/// execution additionally uses it as a specialization/cache dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LaunchClass {
    rows: usize,
    segments: usize,
    demand: Demand,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClassError {
    EmptyRows,
    RowsTooLarge { rows: usize, limit: usize },
    SegmentsTooLarge { segments: usize, limit: usize },
}

impl fmt::Display for ClassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyRows => f.write_str("launch class requires at least one row"),
            Self::RowsTooLarge { rows, limit } => {
                write!(
                    f,
                    "launch class requires {rows} rows but the limit is {limit}"
                )
            }
            Self::SegmentsTooLarge { segments, limit } => write!(
                f,
                "launch class requires {segments} history segments but the limit is {limit}"
            ),
        }
    }
}

impl std::error::Error for ClassError {}

impl LaunchClass {
    /// Round actual launch requirements onto the fixed ladders: rows onto
    /// [`row_class`], segments onto powers of two.
    pub fn covering(
        actual_rows: usize,
        actual_segments: usize,
        demand: Demand,
        row_limit: usize,
    ) -> Result<Self, ClassError> {
        if actual_rows == 0 {
            return Err(ClassError::EmptyRows);
        }
        let limit = row_limit.min(MAX_CLASS_ROWS);
        let rows = row_class(actual_rows)
            .filter(|_| actual_rows <= limit)
            .ok_or(ClassError::RowsTooLarge {
                rows: actual_rows,
                limit,
            })?;
        let actual_segments = actual_segments.max(1);
        let segments = actual_segments
            .checked_next_power_of_two()
            .filter(|segments| *segments <= MAX_CLASS_SEGMENTS)
            .ok_or(ClassError::SegmentsTooLarge {
                segments: actual_segments,
                limit: MAX_CLASS_SEGMENTS,
            })?;
        Ok(Self {
            rows,
            segments,
            demand,
        })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn segments(self) -> usize {
        self.segments
    }

    pub const fn demand(self) -> Demand {
        self.demand
    }
}

impl fmt::Display for LaunchClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "m{}-r{}-d{:x}",
            self.rows,
            self.segments,
            self.demand.bits()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_to_the_fixed_class_ladders() {
        let class = LaunchClass::covering(17, 3, Demand::SELECT, 256).unwrap();
        assert_eq!(class.rows(), 32);
        assert_eq!(class.segments(), 4);
        assert_eq!(class.to_string(), "m32-r4-d4");
    }

    #[test]
    fn prefill_rows_round_to_multiples_of_64() {
        let class = |rows| LaunchClass::covering(rows, 1, Demand::NONE, 512).unwrap().rows();
        assert_eq!(class(33), 64);
        assert_eq!(class(64), 64);
        assert_eq!(class(65), 128);
        assert_eq!(class(129), 192);
        assert_eq!(class(300), 320);
        assert_eq!(class(449), 512);
        assert_eq!(class(512), 512);
        assert_eq!(class(3), 4);
    }

    #[test]
    fn the_row_ladder_is_small_powers_then_prefill_multiples() {
        assert_eq!(
            row_classes(512),
            [1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512]
        );
        assert_eq!(row_classes(32), [1, 2, 4, 8, 16, 32]);
        assert_eq!(row_classes(20), [1, 2, 4, 8, 16, 32]);
        assert_eq!(row_classes(100), [1, 2, 4, 8, 16, 32, 64, 128]);
        assert!(row_classes(0).is_empty());
        assert!(row_classes(513).is_empty());
        for actual in 1..=512 {
            let class = row_class(actual).unwrap();
            assert!(class >= actual);
            assert!(row_classes(512).contains(&class));
        }
    }

    #[test]
    fn enforces_runtime_and_protocol_caps() {
        assert_eq!(
            LaunchClass::covering(129, 1, Demand::NONE, 128),
            Err(ClassError::RowsTooLarge {
                rows: 129,
                limit: 128
            })
        );
        assert_eq!(
            LaunchClass::covering(1, 17, Demand::NONE, 512),
            Err(ClassError::SegmentsTooLarge {
                segments: 17,
                limit: 16
            })
        );
    }
}
