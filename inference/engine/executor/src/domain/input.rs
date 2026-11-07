//! input lifecycle for the executor domain.

use super::*;

/// The tokens of a request's sequence, as its target forwards fed them. A
/// forward at `position` feeds its tokens there and drops every token at or
/// after it, so the tokens before any forward's position are the accepted
/// ones, whatever was fed and not accepted (a verification's rejected rows,
/// a discarded lookahead, a rollback). A sequence resumed from another's
/// state starts with that state's tokens (`ResumeState`). It follows the
/// sequence's state as the n-gram token window does on the device.
#[derive(Clone, Debug, Default)]
pub(super) struct TokenHistory(Vec<crate::TokenId>);

impl TokenHistory {
    /// A sequence whose first tokens are `tokens`.
    pub(super) fn starting(tokens: &[crate::TokenId]) -> Self {
        Self(tokens.to_vec())
    }

    /// Feed `tokens` at `position`: the tokens before it are kept, every
    /// one at or after it replaced.
    pub(super) fn feed(&mut self, position: usize, tokens: &[crate::TokenId]) -> Result<(), String> {
        if position > self.0.len() {
            return Err(format!(
                "a forward at {position} follows {} fed tokens",
                self.0.len()
            ));
        }
        self.0.truncate(position);
        self.0.extend_from_slice(tokens);
        Ok(())
    }

    /// The tokens before `position`.
    pub(super) fn before(&self, position: usize) -> Result<&[crate::TokenId], String> {
        self.0.get(..position).ok_or_else(|| {
            format!(
                "the sequence holds {} tokens, not every one before {position}",
                self.0.len()
            )
        })
    }
}

impl<F: ProgramFamily> ExecutorDomain<F> {
    /// Install a request's prepared input for its lifetime. It is not
    /// resident until [`Self::open_state`]; its images are encoded then.
    pub fn install_input(
        &mut self,
        request: RequestId,
        input: PreparedModelInput,
    ) -> Result<(), String> {
        if self.input.contains_key(&request)
            || self.target.contains_key(&request)
            || self.head.contains_key(&request)
        {
            return Err(format!("request {} is already open", request.0));
        }
        if input.vision().len() > self.execution.policy().limits().max_images_per_request {
            return Err("input exceeds planned image capacity".into());
        }
        if self.definition.vision.is_some() {
            for prepared in input.vision().values() {
                self.image_class(prepared.spatial().rows())?;
            }
        }
        let images = input
            .vision()
            .iter()
            .map(|(identity, prepared)| {
                let image = ImageRef::prepared(self.domain.id().clone(), prepared.clone())
                    .map_err(|error| error.to_string())?;
                Ok((
                    identity.clone(),
                    InputImage {
                        image,
                        features: None,
                    },
                ))
            })
            .collect::<Result<_, String>>()?;
        self.input.insert(
            request,
            RequestInput {
                boundary: None,
                input,
                images,
                resident: false,
                tokens: TokenHistory::default(),
            },
        );
        Ok(())
    }

    /// The tokens of `request`'s sequence before `position`, at most
    /// `count`, oldest first: the accepted tokens a forward at `position`
    /// follows. The host's form of the n-gram token window: the host
    /// evaluates the n-gram row rule (`NGramHash::rows`) over them.
    pub(crate) fn preceding_tokens(
        &self,
        request: RequestId,
        position: usize,
        count: usize,
    ) -> Result<&[crate::TokenId], String> {
        let before = self.installed_input(request)?.tokens.before(position)?;
        Ok(&before[before.len().saturating_sub(count)..])
    }

    /// Prefetch the stored-table rows `request`'s sequence will read when
    /// its target feeds `tokens` (host-known: a prompt, resumed rows) at
    /// `position`, following the tokens it holds before it: each n-gram
    /// input's row rule (`NGramHash::rows`) over them, the rows its steps
    /// will name, held for the request until they are read or it ends
    /// (`StoredTable::prefetch`), leaving one launch's rows of each table
    /// unheld. What a waiter for the prefetches waits on; the domain's work
    /// does not.
    pub fn prefetch_prompt(
        &self,
        request: RequestId,
        position: usize,
        tokens: &[crate::TokenId],
    ) -> Result<Vec<crate::stored_table::Prefetched>, String> {
        let tables = self.family.stored_tables()?;
        let launch = self.execution.policy().limits().max_launch_rows;
        let tokens = tokens.iter().map(|token| u64::from(token.0)).collect::<Vec<_>>();
        crate::operators::row_rules(&self.definition.decoder)
            .into_iter()
            .filter_map(|(block, hash)| tables.get(&block).map(|table| (table, hash)))
            .map(|(table, hash)| {
                let preceding = self
                    .preceding_tokens(request, position, hash.context as usize)?
                    .iter()
                    .map(|token| u64::from(token.0))
                    .collect::<Vec<_>>();
                let heads = usize::try_from(hash.heads().map_err(|error| error.to_string())?)
                    .map_err(|_| "an n-gram input's heads exceed the host's")?;
                let reserve = heads.checked_mul(launch).ok_or("a launch's table rows overflow")?;
                table.prefetch(request.0, hash.rows(&preceding, &tokens), reserve)
            })
            .collect()
    }

    /// The prefetches of `requests`, which ended, are no longer held
    /// (`StoredTable::release`).
    pub(super) fn release_prefetches(&self, requests: &[RequestId]) -> Result<(), String> {
        for table in self.family.stored_tables()?.values() {
            for request in requests {
                table.release(request.0)?;
            }
        }
        Ok(())
    }

    /// What each stored table's steps read after they were queued, by the
    /// block whose n-gram input reads it (`StoredTable::waits`).
    pub fn stored_table_waits(&self) -> Result<Vec<(usize, crate::stored_table::TableWaits)>, String> {
        Ok(self.family.stored_tables()?.into_iter().map(|(block, table)| (block, table.waits())).collect())
    }

    /// Record that the target group `operations` feeds its tokens.
    pub(super) fn feed_tokens(&mut self, operations: &[Operation]) -> Result<(), String> {
        for operation in operations {
            let Operation::Forward {
                request,
                tokens,
                position,
                ..
            } = operation
            else {
                return Err("non-forward target operation".into());
            };
            self.input
                .get_mut(request)
                .ok_or_else(|| format!("request {} feeds rows without installed input", request.0))?
                .tokens
                .feed(*position, tokens)?;
        }
        Ok(())
    }

    fn installed_input(&self, request: RequestId) -> Result<&RequestInput, String> {
        self.input
            .get(&request)
            .ok_or_else(|| format!("request {} lowers rows without installed input", request.0))
    }

    /// Rotary coordinates of the target rows `position..position + count`,
    /// including continuation rows past the admitted input.
    pub(super) fn input_coordinates(
        &self,
        request: RequestId,
        position: usize,
        count: usize,
    ) -> Result<Vec<[i32; 4]>, String> {
        Ok(self
            .installed_input(request)?
            .input
            .coordinates_at(position, count)
            .map_err(|error| error.to_string())?
            .into_iter()
            .map(|[a, b, c]| [a, b, c, 0])
            .collect())
    }

    pub(super) fn input_rows(
        &self,
        request: RequestId,
        position: usize,
        tokens: &[crate::TokenId],
    ) -> Result<(Vec<[i32; 4]>, Vec<crate::ConditioningSlice>), String> {
        let coordinates = self.input_coordinates(request, position, tokens.len())?;
        let state = self.installed_input(request)?;
        let end = position
            .checked_add(tokens.len())
            .ok_or("input end overflows")?;
        if !state.input.layout().boundary(position)
            || !state.input.layout().boundary(end)
            || (position < state.input.tokens().len() && end > state.input.tokens().len())
            || tokens.iter().enumerate().any(|(index, token)| {
                state
                    .input
                    .tokens()
                    .get(position + index)
                    .is_some_and(|expected| expected != token)
            })
        {
            return Err("target rows differ from admitted prepared input".into());
        }
        let mut slices = Vec::new();
        for span in state
            .input
            .layout()
            .spans()
            .iter()
            .filter(|span| span.start < end && position < span.end)
        {
            let image = state
                .images
                .get(&span.identity)
                .ok_or("prepared input image is absent")?;
            let features = image
                .features
                .clone()
                .ok_or("prepared input image has not been encoded")?;
            let start = position.max(span.start);
            let stop = end.min(span.end);
            slices.push(crate::ConditioningSlice {
                source: FeatureSpan::new(features, start - span.start, stop - start)
                    .map_err(|error| error.to_string())?,
                destination: start - position,
            });
        }
        Ok((coordinates, slices))
    }
}
