use super::WeightPlan;
use std::ops::Range;

/// The source-storage schedule and admission share this bound. A batch holds
/// one mapping plus its staged uploads; their sum cannot exceed the largest
/// planned source, plus the mapping's page rounding. Streamed tables contribute
/// upload slots, never their complete stored extent.
pub(crate) struct ImportPlan {
    source_bytes: u64,
}

/// The actual backing an import uses, selected by ImportSource::reads_in_place.
#[derive(Clone)]
pub(crate) enum ImportBacking {
    Mapped(Range<u64>),
    Staged(u64),
}

/// One ordered submission's sources. Only the mapped subset needs a window.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ImportBatch {
    pub(crate) len: usize,
    pub(crate) mapping: Option<Range<u64>>,
    staged_bytes: u64,
}

impl ImportBatch {
    fn include(&self, backing: ImportBacking) -> Option<Self> {
        let (mapping, staged_bytes) = match backing {
            ImportBacking::Mapped(range) => (
                Some(match &self.mapping {
                    Some(current) => current.start.min(range.start)..current.end.max(range.end),
                    None => range,
                }),
                self.staged_bytes,
            ),
            ImportBacking::Staged(bytes) => {
                (self.mapping.clone(), self.staged_bytes.checked_add(bytes)?)
            }
        };
        Some(Self {
            len: self.len + 1,
            mapping,
            staged_bytes,
        })
    }

    fn source_bytes(&self) -> Option<u64> {
        self.staged_bytes.checked_add(
            self.mapping
                .as_ref()
                .map_or(0, |range| range.end - range.start),
        )
    }
}

impl ImportPlan {
    pub(crate) fn new<'a>(weights: impl IntoIterator<Item = &'a WeightPlan>) -> Self {
        Self {
            source_bytes: weights
                .into_iter()
                .map(|weight| weight.source_bytes)
                .max()
                .unwrap_or(0),
        }
    }

    /// Bound the page-rounded mapping even before the file offsets are known.
    /// A staged upload uses no more than the unrounded source bound.
    pub(crate) fn peak_bytes(&self) -> Result<u64, String> {
        if self.source_bytes == 0 {
            return Ok(0);
        }
        #[cfg(unix)]
        {
            let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            let page = u64::try_from(page)
                .map_err(|_| "host page size is unavailable for import planning")?;
            self.source_bytes
                .checked_add(page.checked_mul(2).ok_or("import page bound overflow")?)
                .ok_or_else(|| "import window bound overflow".into())
        }
        #[cfg(not(unix))]
        {
            Ok(self.source_bytes)
        }
    }

    /// Schedule a prefix of planned sources from one file. Each source fits
    /// individually by its WeightPlan. Further sources join only if the
    /// mapping (including gaps) plus all live uploads fit the same bound.
    pub(crate) fn batch<K: PartialEq>(
        &self,
        sources: impl IntoIterator<Item = (K, ImportBacking)>,
    ) -> Option<ImportBatch> {
        let mut sources = sources.into_iter();
        let (file, first) = sources.next()?;
        let (mapping, staged_bytes) = match first {
            ImportBacking::Mapped(range) => (Some(range), 0),
            ImportBacking::Staged(bytes) => (None, bytes),
        };
        let mut batch = ImportBatch {
            len: 1,
            mapping,
            staged_bytes,
        };
        for (next_file, backing) in sources {
            if next_file != file {
                break;
            }
            let Some(next) = batch.include(backing) else {
                break;
            };
            if !next
                .source_bytes()
                .is_some_and(|bytes| bytes <= self.source_bytes)
            {
                break;
            }
            batch = next;
        }
        Some(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapped_batches_bound_spans_including_gaps_and_stop_at_shards() {
        let plan = ImportPlan { source_bytes: 100 };
        let ranges = [(0, 13..53), (0, 63..113), (0, 113..114), (1, 0..100)];
        let sources = || {
            ranges
                .iter()
                .map(|(file, range)| (*file, ImportBacking::Mapped(range.clone())))
        };
        let batch = plan.batch(sources()).unwrap();
        assert_eq!(batch.len, 2);
        assert_eq!(batch.mapping, Some(13..113));
        assert_eq!(plan.batch(sources().skip(2)).unwrap().len, 1);
        assert_eq!(plan.batch(sources().skip(3)).unwrap().len, 1);
        assert!(
            plan.batch(std::iter::empty::<(u8, ImportBacking)>())
                .is_none()
        );
    }

    #[test]
    fn uploads_and_mapping_share_the_bound_without_mapping_staged_ranges() {
        let plan = ImportPlan { source_bytes: 100 };
        let batch = plan
            .batch([
                (0, ImportBacking::Staged(40)),
                (0, ImportBacking::Mapped(1000..1040)),
                (0, ImportBacking::Staged(20)),
                (0, ImportBacking::Staged(1)),
            ])
            .unwrap();
        assert_eq!(batch.len, 3);
        assert_eq!(batch.mapping, Some(1000..1040));
        assert_eq!(batch.source_bytes(), Some(100));
        let batch = plan
            .batch([
                (0, ImportBacking::Staged(60)),
                (0, ImportBacking::Staged(41)),
            ])
            .unwrap();
        assert_eq!(batch.len, 1);
        assert_eq!(batch.mapping, None);
    }

    #[cfg(unix)]
    #[test]
    fn page_bound_covers_every_offset_and_source_length() {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        for source_bytes in [1, page - 1, page, page + 1, 100 * page] {
            let plan = ImportPlan { source_bytes };
            let peak = plan.peak_bytes().unwrap();
            for offset in 0..page {
                let mapped = (offset + source_bytes).div_ceil(page) * page;
                assert!(mapped <= peak);
            }
        }
        assert_eq!(ImportPlan { source_bytes: 0 }.peak_bytes().unwrap(), 0);
    }
}
