use crate::Demand;
use std::fmt;

pub const MAX_CLASS_ROWS: usize = 512;
pub const MAX_CLASS_SEGMENTS: usize = 16;

/// Physical preparation class shared by native and planned execution paths.
///
/// Native Metal uses this as launch shape and qualification data; planned
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
    /// Round actual launch requirements onto the fixed power-of-two ladders.
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
        let rows = actual_rows
            .checked_next_power_of_two()
            .filter(|rows| *rows <= limit)
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
