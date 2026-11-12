use super::WeightPlan;

/// The source-storage schedule and admission share this bound. A batch holds
/// its weights' staged uploads; their sum cannot exceed the largest planned
/// source. Streamed tables contribute upload slots, never their complete
/// stored extent.
pub(crate) struct ImportPlan {
    source_bytes: u64,
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

    /// The most source storage a batch holds at once.
    pub(crate) fn peak_bytes(&self) -> u64 {
        self.source_bytes
    }

    /// How many of `sources` (each a source's file and staged bytes), a
    /// prefix from one file, one batch imports: each source fits by its
    /// WeightPlan, and further sources join while all of the batch's uploads
    /// fit the same bound. None for no sources.
    pub(crate) fn batch<K: PartialEq>(&self, sources: impl IntoIterator<Item = (K, u64)>) -> Option<usize> {
        let mut sources = sources.into_iter();
        let (file, mut staged) = sources.next()?;
        let mut len = 1;
        for (next_file, bytes) in sources {
            match staged.checked_add(bytes) {
                Some(next) if next_file == file && next <= self.source_bytes => staged = next,
                _ => break,
            }
            len += 1;
        }
        Some(len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_share_the_bound_and_stop_at_files() {
        let plan = ImportPlan { source_bytes: 100 };
        assert_eq!(plan.batch([(0, 40), (0, 40), (0, 20), (0, 1)]), Some(3));
        assert_eq!(plan.batch([(0, 60), (0, 41)]), Some(1));
        assert_eq!(plan.batch([(0, 40), (1, 40)]), Some(1));
        assert_eq!(plan.batch(std::iter::empty::<(u8, u64)>()), None);
    }
}
