use super::{next_identity, Generation};
use crate::{models::sequence::OwnedSequence, state::StateCheckpoint};

/// Reconciled request continuation. Numerical storage stays shared and pinned;
/// each fork receives independent grammar, output, and logical progress.
pub struct Checkpoint {
    generation: Generation,
    numerical: StateCheckpoint,
}
impl Checkpoint {
    pub fn fork(&self) -> Result<(Generation, OwnedSequence), String> {
        let generation = self.generation.fork_reconciled()?;
        Ok((generation, OwnedSequence::new(self.numerical.fork())))
    }
    pub fn position(&self) -> usize {
        self.numerical.position()
    }
}
impl Generation {
    /// The execution owner must supply this request's associated sequence.
    /// Neither a completed-but-unaccepted advance nor an evicted logical record
    /// can be captured as a reconciled numerical continuation.
    pub fn checkpoint(&self, sequence: &OwnedSequence) -> Result<Checkpoint, String> {
        if self.pending.is_some() || !self.resident {
            return Err("generation checkpoint requires reconciled resident state".into());
        }
        let numerical = sequence.checkpoint()?;
        if numerical.position() != self.processed {
            return Err("checkpoint numerical and generation positions differ".into());
        }
        Ok(Checkpoint {
            generation: self.fork_reconciled()?,
            numerical,
        })
    }

    pub(crate) fn fork_at(&self, numerical_position: usize) -> Result<Self, String> {
        if self.pending.is_some() || !self.resident || numerical_position != self.processed {
            return Err("checkpoint requires matching reconciled resident state".into());
        }
        self.fork_reconciled()
    }

    fn fork_reconciled(&self) -> Result<Self, String> {
        let constraint = self
            .constraint
            .as_ref()
            .map(|constraint| -> Result<_, String> {
                let fork = constraint.fork();
                if fork.position() != constraint.position() {
                    return Err("checkpoint constraint fork changed its position".into());
                }
                Ok(fork)
            })
            .transpose()?;
        Ok(Self {
            id: next_identity()?,
            revision: 0,
            prompt: self.prompt.clone(),
            layout: self.layout.clone(),
            options: self.options.clone(),
            constraint,
            generated: self.generated.clone(),
            output: self.output.clone(),
            published: self.published,
            processed: self.processed,
            recovery_position: self.recovery_position,
            resident: self.resident,
            finish: self.finish,
            pending: None,
        })
    }
}
